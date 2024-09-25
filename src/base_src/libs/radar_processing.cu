#include "coloradar_tools.h"

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>
#include <cmath>


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

// Host function for generating a Blackman window
void generateBlackmanWindow(std::vector<double>& window, int size) {
    window.resize(size);
    for (int i = 0; i < size; ++i) {
        window[i] = 0.42 - 0.5 * cos(2 * M_PI * i / (size - 1)) + 0.08 * cos(4 * M_PI * i / (size - 1));
    }
}

// Host function for radar cube to heatmap transformation
std::vector<float> cubeToHeatmap(const std::vector<std::complex<double>>& datacube, coloradar::RadarConfig* config) {
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

    // Re-arrange data for angle FFT (rearrangeData is unchanged from the previous code)
    rearrangeData(
        config->numRangeBins, config->numDopplerBins, config->numTxAntennas, config->numRxAntennas,
        config->numAzimuthBins, config->numElevationBins, config->numTxAntennas * config->numRxAntennas,
        config->virtual_array_map, d_windowFuncAzimuth, d_windowFuncElevation, d_datacube, d_datacube
    );
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
