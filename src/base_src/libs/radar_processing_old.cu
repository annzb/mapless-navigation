#include "coloradar_cuda.h"

#include <cuComplex.h>
#include <cufft.h>
#include <cmath>
#include <vector>


__global__ void elementwiseMultiply(int dist, int stride, int n_samples, int n_batches, double* window_func, cuDoubleComplex* data) {
    int sample_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int batch_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (sample_idx < n_samples && batch_idx < n_batches) {
        int data_idx = (batch_idx * dist) + (sample_idx * stride);
        data[data_idx] = cuCmul(make_cuDoubleComplex(window_func[sample_idx], 0.0), data[data_idx]);
    }
}

// CUDA kernel for assembling the message from the processed data
__global__ void assembleOutput(int n_range_bins, int n_doppler_bins, int n_az_beams, int n_el_beams, cuDoubleComplex* data_in, float* data_out) {
    int in_range_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int in_doppler_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int angle_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (in_range_idx < n_range_bins && in_doppler_idx < n_doppler_bins && angle_idx < n_az_beams * n_el_beams) {
        int in_az_idx = int(angle_idx / n_el_beams);
        int in_el_idx = angle_idx - (in_az_idx * n_el_beams);
        int out_doppler_idx = in_doppler_idx < n_doppler_bins / 2 ? in_doppler_idx + (n_doppler_bins / 2) : in_doppler_idx - (n_doppler_bins / 2);
        int out_az_idx = in_az_idx < n_az_beams / 2 ? in_az_idx + (n_az_beams / 2) : in_az_idx - (n_az_beams / 2);
        int out_el_idx = in_el_idx < n_el_beams / 2 ? in_el_idx + (n_el_beams / 2) : in_el_idx - (n_el_beams / 2);
        int in_idx = in_el_idx + n_el_beams * (in_az_idx + n_az_beams * (in_range_idx + n_range_bins * in_doppler_idx));
        int out_idx = in_range_idx + n_range_bins * (out_doppler_idx + n_doppler_bins * (out_az_idx + n_az_beams * out_el_idx));

        double magnitude = sqrt(cuCreal(data_in[in_idx]) * cuCreal(data_in[in_idx]) + cuCimag(data_in[in_idx]) * cuCimag(data_in[in_idx]));
        data_out[out_idx] = static_cast<float>(magnitude);
    }
}

// CUDA kernel for rearranging data based on the virtual array map
__global__ void rearrangeMatrix(int n_range_bins, int n_doppler_bins, int n_tx, int n_rx, int n_az_beams, int n_el_beams, int n_virtual_elements, int* virtual_array_map, cuDoubleComplex* src_mat, cuDoubleComplex* dest_mat) {
    int range_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int doppler_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int virtual_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (range_idx < n_range_bins && doppler_idx < n_doppler_bins && virtual_idx < n_virtual_elements) {
        int az_idx = virtual_array_map[4 * virtual_idx];
        int el_idx = virtual_array_map[(4 * virtual_idx) + 1];
        int rx_idx = virtual_array_map[(4 * virtual_idx) + 2];
        int tx_idx = virtual_array_map[(4 * virtual_idx) + 3];

        if (az_idx < n_az_beams && el_idx < n_el_beams) {
            int src_idx = range_idx + n_range_bins * (rx_idx + n_rx * (tx_idx + n_tx * doppler_idx));
            int dest_idx = el_idx + n_el_beams * (az_idx + n_az_beams * (range_idx + n_range_bins * doppler_idx));

            dest_mat[dest_idx] = src_mat[src_idx];
        }
    }
}

// Host function for rearranging data using virtual array map
void rearrangeData(int num_range_bins, int num_doppler_bins, int num_tx, int num_rx, int num_az_beams, int num_el_beams, int num_virtual_elements, int* virtual_array_map, cuDoubleComplex* src_mat, cuDoubleComplex* dest_mat) {
    dim3 threads_per_block(8, 8, 8);
    dim3 num_blocks((num_range_bins + threads_per_block.x - 1) / threads_per_block.x,
                    (num_doppler_bins + threads_per_block.y - 1) / threads_per_block.y,
                    (num_virtual_elements + threads_per_block.z - 1) / threads_per_block.z);

    rearrangeMatrix<<<num_blocks, threads_per_block>>>(num_range_bins, num_doppler_bins, num_tx, num_rx, num_az_beams, num_el_beams, num_virtual_elements, virtual_array_map, src_mat, dest_mat);
    cudaDeviceSynchronize();
}

void applyWindow(int dist,
                 int stride,
                 int n_samples,
                 int n_batches,
                 double* window_func,
                 cuDoubleComplex* data)
{
  dim3 threads_per_block(16,16);
  dim3 num_blocks(n_samples / threads_per_block.x,
                 n_batches / threads_per_block.y);
  elementwiseMultiply<<<num_blocks,threads_per_block>>>(dist, stride, n_samples, n_batches, window_func, data);
  cudaDeviceSynchronize();
}


// Host function for radar cube to heatmap transformation using old logic
std::vector<float> coloradar::cubeToHeatmap(const std::vector<std::complex<double>>& datacube, coloradar::RadarConfig* config) {
    std::vector<float> heatmap;
    int totalElements = config->numRangeBins * config->numDopplerBins * config->numAzimuthBins * config->numElevationBins;
    heatmap.resize(totalElements);

    // Allocate device memory
    cuDoubleComplex* d_datacube;
    float* d_heatmap;
    cudaMalloc(&d_datacube, datacube.size() * sizeof(cuDoubleComplex));
    cudaMalloc(&d_heatmap, totalElements * sizeof(float));

    // Copy the datacube to the GPU
    cudaMemcpy(d_datacube, datacube.data(), datacube.size() * sizeof(cuDoubleComplex), cudaMemcpyHostToDevice);

    // Apply range window function
    applyWindow(config->numRangeBins, 1, config->numRangeBins, config->numTxAntennas * config->numRxAntennas * config->numDopplerBins, config->rangeWindowFunc, d_datacube);

    // Execute the range FFT using cuFFT
    cufftHandle range_plan;
    cufftPlan1d(&range_plan, config->numRangeBins, CUFFT_Z2Z, config->numTxAntennas * config->numRxAntennas * config->numDopplerBins);
    cufftExecZ2Z(range_plan, d_datacube, d_datacube, CUFFT_FORWARD);
    cufftDestroy(range_plan);

    // Apply Doppler window function
    applyWindow(config->numDopplerBins, config->numRangeBins * config->numTxAntennas * config->numRxAntennas, config->numDopplerBins, config->numTxAntennas * config->numRxAntennas, config->dopplerWindowFunc, d_datacube);

    // Execute the Doppler FFT using cuFFT
    cufftHandle doppler_plan;
    cufftPlan1d(&doppler_plan, config->numDopplerBins, CUFFT_Z2Z, config->numTxAntennas * config->numRxAntennas);
    cufftExecZ2Z(doppler_plan, d_datacube, d_datacube, CUFFT_FORWARD);
    cufftDestroy(doppler_plan);

    // Rearrange data based on virtual array map
    rearrangeData(config->numRangeBins, config->numDopplerBins, config->numTxAntennas, config->numRxAntennas, config->numAzimuthBins, config->numElevationBins, config->numVirtualElements, config->virtualArrayMap, d_datacube, d_datacube);

    // Perform the angle FFT using cuFFT
    cufftHandle angle_plan;
    int n[2] = {config->numAzimuthBins, config->numElevationBins};
    cufftPlanMany(&angle_plan, 2, n, NULL, 1, config->numAzimuthBins * config->numElevationBins, NULL, 1, config->numAzimuthBins * config->numElevationBins, CUFFT_Z2Z, config->numRangeBins * config->numDopplerBins);
    cufftExecZ2Z(angle_plan, d_datacube, d_datacube, CUFFT_FORWARD);
    cufftDestroy(angle_plan);

    // Assemble the final heatmap message
    assembleOutput<<<(totalElements + 255) / 256, 256>>>(config->numRangeBins, config->numDopplerBins, config->numAzimuthBins, config->numElevationBins, d_datacube, d_heatmap);
    cudaDeviceSynchronize();

    // Copy the heatmap back to host memory
    cudaMemcpy(heatmap.data(), d_heatmap, totalElements * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up device memory
    cudaFree(d_datacube);
    cudaFree(d_heatmap);

    return heatmap;
}
