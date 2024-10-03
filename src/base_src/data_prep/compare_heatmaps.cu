#include <iostream>
#include "coloradar_cuda.h"
#include <set>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <iomanip>

namespace fs = std::filesystem;


template<typename T>
void cudaCopy(T* dest, std::vector<T> source) {
    cudaError_t err = cudaMemcpy(dest, source.data(), sizeof(T) * source.size(), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        std::cerr << "CUDA memcpy error:  " << cudaGetErrorString(err) << std::endl;
    }
}


float roundToTwoDecimals(float value) {
    return std::round(value * 10000.0) / 10000.0;
}

bool compareVectorsIgnoringOrder(const std::vector<float>& vec1, const std::vector<float>& vec2) {
    std::multiset<float> set1, set2;
    for (const auto& val : vec1) set1.insert(roundToTwoDecimals(val));
    for (const auto& val : vec2) set2.insert(roundToTwoDecimals(val));
    std::unordered_set<float> unique_elements(set1.begin(), set1.end());
    unique_elements.insert(set2.begin(), set2.end());
    size_t match_count = 0;
    for (const auto& val : unique_elements) {
        size_t count1 = set1.count(val);
        size_t count2 = set2.count(val);
        match_count += std::min(count1, count2);
    }
    size_t total_elements = vec1.size();
    size_t mismatch_count = total_elements - match_count;
    double mismatch_percentage = static_cast<double>(mismatch_count) / total_elements * 100.0;
    std::cout << "Number of mismatched elements without ordering: " << std::fixed << std::setprecision(2) << mismatch_percentage << "%" << std::endl;
    return mismatch_count == 0;
}


std::vector<cuDoubleComplex> toCudaComplex(std::vector<std::complex<double>> array) {
    std::vector<cuDoubleComplex> cudaArray(array.size());
    for (size_t i = 0; i < array.size(); ++i) {
        cudaArray[i] = make_cuDoubleComplex(array[i].real(), array[i].imag());
    }
    return cudaArray;
}

__global__
void expandDoppler(int n_range_bins,
                   int n_doppler_bins,
                   int n_angles,
                   float doppler_bin_width,
                   float* data_in,
                   float* data_out_big)
{
  int range_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int angle_idx = blockIdx.y * blockDim.y + threadIdx.y;

  if (range_idx < n_range_bins && angle_idx < n_angles) {

    // Extract data from small array
    int out_idx = 2 * (range_idx + n_range_bins * angle_idx);
    float intensity = data_in[out_idx];   // Intensity from small array
    float doppler_shift = data_in[out_idx + 1]; // Doppler shift from small array

    // Calculate doppler index (activation center) from the small array
    int activation_center = (doppler_shift / doppler_bin_width) + (n_doppler_bins / 2);

    // Set all elements of the big array to zero
    for (int doppler_idx = 0; doppler_idx < n_doppler_bins; doppler_idx++) {
      int idx = range_idx + n_range_bins * doppler_idx + n_range_bins * n_doppler_bins * angle_idx;
      data_out_big[idx] = 0.0f;  // Set sparse entries to zero
    }

    // Place the intensity value at the corresponding doppler index
    if (activation_center >= 0 && activation_center < n_doppler_bins) {
      int idx = range_idx + n_range_bins * activation_center
                + n_range_bins * n_doppler_bins * angle_idx;
      data_out_big[idx] = intensity;
    }
  }
}


void applyExpandDoppler(int n_range_bins,
                   int n_doppler_bins,
                   int n_angles,
                   float doppler_bin_width,
                   float* data_in,
                   float* data_out_big)
{
  dim3 threads(16, 16);
  dim3 blocks((n_range_bins + threads.x - 1) / threads.x,
              (n_angles + threads.y - 1) / threads.y);

  expandDoppler<<<blocks, threads>>>(n_range_bins,
                                     n_doppler_bins,
                                     n_angles,
                                     doppler_bin_width,
                                     data_in,
                                     data_out_big);
  cudaDeviceSynchronize();
}


std::vector<float> reconstructCollapsedHeatmap(std::vector<float> heatmapSmall, coloradar::RadarConfig* config) {
    float* heatmapSmallGpu;
    float* heatmapBigGpu;
    int expandedHeatmapSize = config->numAngles * config->numPosRangeBins * config->numDopplerBins;

    cudaMalloc(&heatmapSmallGpu, sizeof(float) * heatmapSmall.size());
    cudaMalloc(&heatmapBigGpu, sizeof(float) * expandedHeatmapSize);
    cudaCopy(heatmapSmallGpu, heatmapSmall);
    applyExpandDoppler(config->numPosRangeBins, config->numDopplerBins, config->numAngles, config->dopplerBinWidth, heatmapSmallGpu, heatmapBigGpu);

    std::vector<float> heatmapBig(expandedHeatmapSize);
    cudaMemcpy(&heatmapBig[0], heatmapBigGpu, sizeof(float) * expandedHeatmapSize, cudaMemcpyDefault);
    cudaFree(heatmapSmallGpu);
    cudaFree(heatmapBigGpu);
    return heatmapBig;
}



// std::vector<float> applyCouplingCalib(std::vector<float> heatmap, coloradar::RadarConfig* config) {
//     float* heatmapGpu;
//     cuDoubleComplex* couplingSignatureGpu;
//     std::vector<cuDoubleComplex> couplingSignature = toCudaComplex(config->couplingCalibMatrix);
//
//     cudaMalloc(&heatmapGpu, sizeof(float) * heatmap.size());
//     cudaMalloc(&couplingSignatureGpu, sizeof(cuDoubleComplex) * couplingSignature.size());
//     cudaCopy(couplingSignatureGpu, couplingSignature);
//     cudaCopy(heatmapGpu, heatmap);
//     removeCoupling(config->numPosRangeBins, config->numDopplerBins, config->numTxAntennas * config->numRxAntennas, heatmapGpu, couplingSignatureGpu);
//
//     std::vector<float> heatmapCalibrated;
//     cudaMemcpy(&heatmapCalibrated[0], heatmapGpu, sizeof(float) * heatmap.size(), cudaMemcpyDefault);
//     cudaFree(heatmapGpu);
//     cudaFree(couplingSignatureGpu);
//     return heatmapCalibrated;
// }
//
// std::vector<float> applyPhaseFreqCalib(std::vector<float> heatmap, coloradar::RadarConfig* config) {
//     float* heatmapGpu;
//     cuDoubleComplex* phaseCalibMatrixGpu;
//     cuDoubleComplex* freqCalibMatrixGpu;
//     std::vector<cuDoubleComplex> phaseCalibMatrix = toCudaComplex(config->calPhaseCalibMatrix);
//     std::vector<cuDoubleComplex> freqCalibMatrix = toCudaComplex(config->calFrequencyCalibMatrix);
//
//     cudaMalloc(&heatmapGpu, sizeof(float) * heatmap.size());
//     cudaMalloc(&phaseCalibMatrixGpu, sizeof(cuDoubleComplex) * phaseCalibMatrix.size());
//     cudaMalloc(&freqCalibMatrixGpu, sizeof(cuDoubleComplex) * freqCalibMatrix.size());
//     cudaCopy(heatmapGpu, heatmap);
//     cudaCopy(phaseCalibMatrixGpu, phaseCalibMatrix);
//     cudaCopy(freqCalibMatrixGpu, freqCalibMatrix);
//     applyPhaseFreqCal(config->numRangeBins, config->numDopplerBins, config->numTxAntennas, config->numRxAntennas, heatmapGpu, freqCalibMatrixGpu, phaseCalibMatrixGpu);
//
//     std::vector<float> heatmapCalibrated;
//     cudaMemcpy(&heatmapCalibrated[0], heatmapGpu, sizeof(float) * heatmap.size(), cudaMemcpyDefault);
//     cudaFree(heatmapGpu);
//     cudaFree(phaseCalibMatrixGpu);
//     cudaFree(freqCalibMatrixGpu);
//     return heatmapCalibrated;
// }


int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <coloradar_dir> <run_name>" << std::endl;
        return 1;
    }
    fs::path coloradarDir = argv[1];
    std::string runName = argv[2];

    coloradar::ColoradarDataset dataset(coloradarDir);
    coloradar::ColoradarRun run = dataset.getRun(runName);

    std::vector<int16_t> datacube = run.getDatacube(0, &dataset.cascadeConfig);
    std::cout << "Read cube of size " << datacube.size() << std::endl;

    std::vector<float> heatmap = run.getHeatmap(0, &dataset.cascadeConfig);
    std::cout << "Read heatmap of size " << heatmap.size() << std::endl;

//     trueHeatmap = applyCouplingCalib(trueHeatmap, &dataset.cascadeConfig);
//     std::cout << "Applied coupling calibration on true heatmap" << std::endl;
//     trueHeatmap = applyPhaseFreqCalib(trueHeatmap, &dataset.cascadeConfig);
//     std::cout << "Applied phase frequency calibration on true heatmap" << std::endl;

//     float* heatmapGPU;
//     float* heatmapGPUCollapsed;
//     int collapsedHeatmapSize = 2 * dataset.cascadeConfig.numPosRangeBins * dataset.cascadeConfig.numAzimuthBeams * dataset.cascadeConfig.numElevationBeams;
//     cudaMalloc(&heatmapGPU, sizeof(float) * trueHeatmap.size());
//     cudaMalloc(&heatmapGPUCollapsed, sizeof(float) * collapsedHeatmapSize);
//     cudaCopy(heatmapGPU, trueHeatmap);
//     collapseDoppler(dataset.cascadeConfig.numPosRangeBins, dataset.cascadeConfig.numDopplerBins, dataset.cascadeConfig.numAngles, dataset.cascadeConfig.dopplerBinWidth, heatmapGPU, heatmapGPUCollapsed);
//     std::vector<float> heatmap(collapsedHeatmapSize);
//     cudaMemcpy(&heatmap[0], heatmapGPUCollapsed, sizeof(float) * collapsedHeatmapSize, cudaMemcpyDefault);
//     cudaFree(heatmapGPU);
//     cudaFree(heatmapGPUCollapsed);
//     std::cout << "Collapsed true heatmap into " << heatmap.size() << std::endl;

    std::vector<float> computedHeatmap = coloradar::cubeToHeatmap(datacube, &dataset.cascadeConfig);
    std::cout << "Computed heatmap of size " << computedHeatmap.size() << std::endl;

//     std::vector<float> computedHeatmap = reconstructCollapsedHeatmap(computedHeatmapInitial, &dataset.cascadeConfig);
//     std::cout << "Expanded computed heatmap into " << computedHeatmap.size() << std::endl;

    bool match = true;
    float threshold = 1e-4;
    if (computedHeatmap.size() != heatmap.size()) {
        std::cerr << "Error: Size mismatch between computed and actual heatmap!" << std::endl;
        return 1;
    }
    float mismatchCount = 0, computedNonZeroCount = 0, trueNonZeroCount = 0;
    for (size_t i = 0; i < computedHeatmap.size(); ++i) {
        if (heatmap[i] != 0.0) trueNonZeroCount++;
        if (computedHeatmap[i] != 0.0) computedNonZeroCount++;
        if (std::abs(computedHeatmap[i] - heatmap[i]) > threshold) {
            mismatchCount++;
            // std::cout << "Mismatch at index " << i << ": computed = " << computedHeatmap[i] << ", actual = " << heatmap[i] << std::endl;
        }
    }
    if (match && mismatchCount == 0) {
        std::cout << "Success! The computed heatmap matches the actual heatmap." << std::endl;
    } else {
        std::cout << "The computed heatmap does not match the actual heatmap. Number of mismatched elements: " << mismatchCount << " (" << mismatchCount / computedHeatmap.size() * 100 << "%)" << std::endl;
        std::cout << "Non-zero elements in true heatmap: " << trueNonZeroCount << " (" << trueNonZeroCount / heatmap.size() * 100 << " %)" << std::endl;
        std::cout << "Non-zero elements in computed heatmap: " << computedNonZeroCount << " (" << computedNonZeroCount / computedHeatmap.size() * 100 << " %)" << std::endl;
    }

    compareVectorsIgnoringOrder(heatmap, computedHeatmap);

    return 0;
}
