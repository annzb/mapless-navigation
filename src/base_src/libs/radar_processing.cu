#include "coloradar_cuda.h"

#include <cuComplex.h>
#include <cufft.h>
#include <cmath>
#include <vector>


// CUDA kernel for applying a Blackman window function to the data
__global__ void applyWindowKernel(int dist, int stride, int n_samples, int n_batches, double* window_func, cuDoubleComplex* data) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_samples * n_batches) {
        int sample_idx = (idx / stride) % n_samples;
        data[idx] = cuCmul(data[idx], make_cuDoubleComplex(window_func[sample_idx], 0.0));
    }
}

// CUDA kernel for assembling the message from the processed data
__global__ void assembleMsgKernel(int n_range_bins, int n_doppler_bins, int n_az_beams, int n_el_beams, cuDoubleComplex* data_in, float* data_out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n_range_bins * n_doppler_bins * n_az_beams * n_el_beams;
    if (idx < total_elements) {
        data_out[idx] = cuCabs(data_in[idx]);
    }
}

// CUDA kernel for rearranging data without virtual array map
__global__ void rearrangeMatrixKernel(int n_range_bins, int n_doppler_bins, int n_tx, int n_rx, int n_az_beams, int n_el_beams, int* tx_distance, int* tx_elevation, int* rx_distance, int* rx_elevation, cuDoubleComplex* src_mat, cuDoubleComplex* dest_mat) {
    int range_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int doppler_idx = blockIdx.y * blockDim.y + threadIdx.y;
    int tx_idx = blockIdx.z * blockDim.z + threadIdx.z;

    if (range_idx < n_range_bins && doppler_idx < n_doppler_bins && tx_idx < n_tx) {
        int rx_idx = tx_idx % n_rx;
        int tx_distance_idx = tx_idx / n_rx;

        // Calculate azimuth and elevation based on tx and rx positions
        int az_idx = rx_distance[rx_idx] + tx_distance[tx_distance_idx];
        int el_idx = rx_elevation[rx_idx] + tx_elevation[tx_distance_idx];

        if (az_idx < n_az_beams && el_idx < n_el_beams) {
            int src_idx = range_idx + n_range_bins * (rx_idx + n_rx * (tx_distance_idx + n_tx * doppler_idx));
            int dest_idx = el_idx + n_el_beams * (az_idx + n_az_beams * (range_idx + n_range_bins * doppler_idx));

            dest_mat[dest_idx] = src_mat[src_idx];
        }
    }
}

// Host function for generating a Blackman window
void generateBlackmanWindow(std::vector<double>& window, int size) {
    window.resize(size);
    for (int i = 0; i < size; ++i) {
        window[i] = 0.42 - 0.5 * cos(2 * M_PI * i / (size - 1)) + 0.08 * cos(4 * M_PI * i / (size - 1));
    }
}

// Function to rearrange data using tx and rx distances and elevations
void rearrangeData(int num_range_bins, int num_doppler_bins, int num_tx, int num_rx, int num_az_beams, int num_el_beams, int* tx_distance, int* tx_elevation, int* rx_distance, int* rx_elevation, cuDoubleComplex* src_mat, cuDoubleComplex* dest_mat) {
    dim3 threads_per_block(8, 8, 8);
    dim3 num_blocks((num_range_bins + threads_per_block.x - 1) / threads_per_block.x,
                    (num_doppler_bins + threads_per_block.y - 1) / threads_per_block.y,
                    (num_tx + threads_per_block.z - 1) / threads_per_block.z);

    rearrangeMatrixKernel<<<num_blocks, threads_per_block>>>(num_range_bins, num_doppler_bins, num_tx, num_rx, num_az_beams, num_el_beams, tx_distance, tx_elevation, rx_distance, rx_elevation, src_mat, dest_mat);
    cudaDeviceSynchronize();
}

// Host function for radar cube to heatmap transformation
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

    // Generate and copy window functions to the GPU
    std::vector<double> range_window, doppler_window, az_window, el_window;
    generateBlackmanWindow(range_window, config->numRangeBins);
    generateBlackmanWindow(doppler_window, config->numDopplerBins);
    generateBlackmanWindow(az_window, config->numAzimuthBins);
    generateBlackmanWindow(el_window, config->numElevationBins);

    double* d_windowFuncRange;
    double* d_windowFuncDoppler;
    double* d_windowFuncAzimuth;
    double* d_windowFuncElevation;

    // Consolidate memory allocations and transfers
    cudaMalloc(&d_windowFuncRange, range_window.size() * sizeof(double));
    cudaMemcpy(d_windowFuncRange, range_window.data(), range_window.size() * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_windowFuncDoppler, doppler_window.size() * sizeof(double));
    cudaMemcpy(d_windowFuncDoppler, doppler_window.data(), doppler_window.size() * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_windowFuncAzimuth, az_window.size() * sizeof(double));
    cudaMemcpy(d_windowFuncAzimuth, az_window.data(), az_window.size() * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_windowFuncElevation, el_window.size() * sizeof(double));
    cudaMemcpy(d_windowFuncElevation, el_window.data(), el_window.size() * sizeof(double), cudaMemcpyHostToDevice);

    // Apply the range window function
    int threadsPerBlock = 256;
    int blocksPerGrid = (datacube.size() + threadsPerBlock - 1) / threadsPerBlock;
    applyWindowKernel<<<blocksPerGrid, threadsPerBlock>>>(
        config->numRangeBins, 1, config->numRangeBins, config->numTxAntennas * config->numRxAntennas * config->numDopplerBins, d_windowFuncRange, d_datacube
    );
    cudaDeviceSynchronize();

    // Execute the range FFT using cuFFT
    cufftHandle range_plan;
    cufftPlan1d(&range_plan, config->numRangeBins, CUFFT_Z2Z, config->numTxAntennas * config->numRxAntennas * config->numDopplerBins);
    cufftExecZ2Z(range_plan, d_datacube, d_datacube, CUFFT_FORWARD);
    cufftDestroy(range_plan);

    // Apply the Doppler window function
    applyWindowKernel<<<blocksPerGrid, threadsPerBlock>>>(
        config->numDopplerBins, config->numRangeBins * config->numTxAntennas * config->numRxAntennas, config->numDopplerBins, config->numTxAntennas * config->numRxAntennas, d_windowFuncDoppler, d_datacube
    );
    cudaDeviceSynchronize();

    // Execute the Doppler FFT using cuFFT
    cufftHandle doppler_plan;
    cufftPlan1d(&doppler_plan, config->numDopplerBins, CUFFT_Z2Z, config->numTxAntennas * config->numRxAntennas);
    cufftExecZ2Z(doppler_plan, d_datacube, d_datacube, CUFFT_FORWARD);
    cufftDestroy(doppler_plan);

    // Re-arrange data for angle FFT using tx and rx distances/elevations
    rearrangeData(config->numRangeBins, config->numDopplerBins, config->numTxAntennas, config->numRxAntennas, config->numAzimuthBins, config->numElevationBins, config->txDistance.data(), config->txElevation.data(), config->rxDistance.data(), config->rxElevation.data(), d_datacube, d_datacube);
    cudaDeviceSynchronize();

    // Perform the angle FFT using cuFFT
    cufftHandle angle_plan;
    int n[2] = {config->numAzimuthBins, config->numElevationBins};
    cufftPlanMany(&angle_plan, 2, n, NULL, 1, config->numAzimuthBins * config->numElevationBins, NULL, 1, config->numAzimuthBins * config->numElevationBins, CUFFT_Z2Z, config->numRangeBins * config->numDopplerBins);
    cufftExecZ2Z(angle_plan, d_datacube, d_datacube, CUFFT_FORWARD);
    cufftDestroy(angle_plan);

    // Assemble the message (heatmap)
    assembleMsgKernel<<<blocksPerGrid, threadsPerBlock>>>(
        config->numRangeBins, config->numDopplerBins, config->numAzimuthBins, config->numElevationBins, d_datacube, d_heatmap
    );
    cudaDeviceSynchronize();

    // Copy the heatmap back to host memory
    cudaMemcpy(heatmap.data(), d_heatmap, totalElements * sizeof(float), cudaMemcpyDeviceToHost);

    // Clean up device memory
    cudaFree(d_datacube);
    cudaFree(d_heatmap);
    cudaFree(d_windowFuncRange);
    cudaFree(d_windowFuncDoppler);
    cudaFree(d_windowFuncAzimuth);
    cudaFree(d_windowFuncElevation);

    return heatmap;
}
