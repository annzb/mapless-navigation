#include <iostream>
#include "coloradar_cuda.h"
#include <set>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <iomanip>

namespace fs = std::filesystem;


int calculateIndex5D(int i1, int i2, int i3, int i4, int i5, const std::vector<int>& D) {
    return i1 * (D[1] * D[2] * D[3] * D[4]) + i2 * (D[2] * D[3] * D[4]) + i3 * (D[3] * D[4]) + i4 * D[4] + i5;
}

// Function to rearrange the 1D array based on the permutation vector
std::vector<int16_t> rearrangeArray5D(const std::vector<int16_t>& originalArray, const std::vector<int>& perm) {
    int D1 = 256, D2 = 16, D3 = 16, D4 = 12, D5 = 2;
    std::vector<int> D = {D1, D2, D3, D4, D5};
    std::vector<int16_t> rearrangedArray(originalArray.size());

    for (int i1 = 0; i1 < D1; ++i1) {
        for (int i2 = 0; i2 < D2; ++i2) {
            for (int i3 = 0; i3 < D3; ++i3) {
                for (int i4 = 0; i4 < D4; ++i4) {
                    for (int i5 = 0; i5 < D5; ++i5) {
                        // Mapping current indices to the permutation
                        std::vector<int> idx = {i1, i2, i3, i4, i5};
                        int originalIndex = calculateIndex5D(i1, i2, i3, i4, i5, D);
                        int rearrangedIndex = calculateIndex5D(idx[perm[0]], idx[perm[1]], idx[perm[2]], idx[perm[3]], idx[perm[4]], D);
                        rearrangedArray[rearrangedIndex] = originalArray[originalIndex];
                    }
                }
            }
        }
    }
    return rearrangedArray;
}


int calculateIndex(int i1, int i2, int i3, int i4, int D1, int D2, int D3, int D4) {
    return i1 * (D2 * D3 * D4) + i2 * (D3 * D4) + i3 * D4 + i4;
}

// Function to rearrange the 1D array based on the permutation number
std::vector<float> rearrangeArray(const std::vector<float>& originalArray, int permutation) {
    int D1 = 128, D2 = 128, D3 = 32, D4 = 2;
    std::vector<float> rearrangedArray(originalArray.size());

    for (int i1 = 0; i1 < D1; ++i1) {
        for (int i2 = 0; i2 < D2; ++i2) {
            for (int i3 = 0; i3 < D3; ++i3) {
                for (int i4 = 0; i4 < D4; ++i4) {
                    int originalIndex = calculateIndex(i1, i2, i3, i4, D1, D2, D3, D4);
                    int rearrangedIndex;

                    // Apply the permutation
                    switch (permutation) {
                        case 0:
                            rearrangedIndex = originalIndex;
                        case 1:
                            rearrangedIndex = calculateIndex(i1, i2, i4, i3, D1, D2, D4, D3);
                            break;
                        case 2:
                            rearrangedIndex = calculateIndex(i1, i3, i2, i4, D1, D3, D2, D4);
                            break;
                        case 3:
                            rearrangedIndex = calculateIndex(i1, i3, i4, i2, D1, D3, D4, D2);
                            break;
                        case 4:
                            rearrangedIndex = calculateIndex(i1, i4, i2, i3, D1, D4, D2, D3);
                            break;
                        case 5:
                            rearrangedIndex = calculateIndex(i1, i4, i3, i2, D1, D4, D3, D2);
                            break;
                        case 6:
                            rearrangedIndex = calculateIndex(i2, i1, i3, i4, D2, D1, D3, D4);
                            break;
                        case 7:
                            rearrangedIndex = calculateIndex(i2, i1, i4, i3, D2, D1, D4, D3);
                            break;
                        case 8:
                            rearrangedIndex = calculateIndex(i2, i3, i1, i4, D2, D3, D1, D4);
                            break;
                        case 9:
                            rearrangedIndex = calculateIndex(i2, i3, i4, i1, D2, D3, D4, D1);
                            break;
                        case 10:
                            rearrangedIndex = calculateIndex(i2, i4, i1, i3, D2, D4, D1, D3);
                            break;
                        case 11:
                            rearrangedIndex = calculateIndex(i2, i4, i3, i1, D2, D4, D3, D1);
                            break;
                        case 12:
                            rearrangedIndex = calculateIndex(i3, i1, i2, i4, D3, D1, D2, D4);
                            break;
                        case 13:
                            rearrangedIndex = calculateIndex(i3, i1, i4, i2, D3, D1, D4, D2);
                            break;
                        case 14:
                            rearrangedIndex = calculateIndex(i3, i2, i1, i4, D3, D2, D1, D4);
                            break;
                        case 15:
                            rearrangedIndex = calculateIndex(i3, i2, i4, i1, D3, D2, D4, D1);
                            break;
                        case 16:
                            rearrangedIndex = calculateIndex(i3, i4, i1, i2, D3, D4, D1, D2);
                            break;
                        case 17:
                            rearrangedIndex = calculateIndex(i3, i4, i2, i1, D3, D4, D2, D1);
                            break;
                        case 18:
                            rearrangedIndex = calculateIndex(i4, i1, i2, i3, D4, D1, D2, D3);
                            break;
                        case 19:
                            rearrangedIndex = calculateIndex(i4, i1, i3, i2, D4, D1, D3, D2);
                            break;
                        case 20:
                            rearrangedIndex = calculateIndex(i4, i2, i1, i3, D4, D2, D1, D3);
                            break;
                        case 21:
                            rearrangedIndex = calculateIndex(i4, i2, i3, i1, D4, D2, D3, D1);
                            break;
                        case 22:
                            rearrangedIndex = calculateIndex(i4, i3, i1, i2, D4, D3, D1, D2);
                            break;
                        case 23:
                            rearrangedIndex = calculateIndex(i4, i3, i2, i1, D4, D3, D2, D1);
                            break;
                        default:
                            rearrangedIndex = originalIndex; // Original order
                            break;
                    }

                    rearrangedArray[rearrangedIndex] = originalArray[originalIndex];
                }
            }
        }
    }
    return rearrangedArray;
}


float roundToTwoDecimals(float value, double roundMultiplier) {
    return std::round(value * roundMultiplier) / roundMultiplier;
}

bool compareVectorsIgnoringOrder(const std::vector<float>& vec1, const std::vector<float>& vec2, int decimalAccuracy = 0) {
    double roundMultiplier = std::pow(10, decimalAccuracy);
    std::multiset<float> set1, set2;
    for (const auto& val : vec1) set1.insert(roundToTwoDecimals(val, roundMultiplier));
    for (const auto& val : vec2) set2.insert(roundToTwoDecimals(val, roundMultiplier));
    std::unordered_set<float> unique_elements(set1.begin(), set1.end());
    unique_elements.insert(set2.begin(), set2.end());
    size_t match_count = 0;
    for (const auto& val : unique_elements) {
        size_t count1 = set1.count(val);
        size_t count2 = set2.count(val);
        match_count += std::min(count1, count2);
    }
    size_t total_elements = vec1.size();
    double mismatch_percentage = static_cast<double>(match_count) / total_elements * 100.0;
    std::cout << "Number of matched elements without ordering: " << std::fixed << std::setprecision(2) << mismatch_percentage << "%" << std::endl;
    return total_elements - match_count == 0;
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
    coloradar::cudaCopy(heatmapSmallGpu, heatmapSmall);
    applyExpandDoppler(config->numPosRangeBins, config->numDopplerBins, config->numAngles, config->dopplerBinWidth, heatmapSmallGpu, heatmapBigGpu);

    std::vector<float> heatmapBig(expandedHeatmapSize);
    cudaMemcpy(&heatmapBig[0], heatmapBigGpu, sizeof(float) * expandedHeatmapSize, cudaMemcpyDefault);
    cudaFree(heatmapSmallGpu);
    cudaFree(heatmapBigGpu);
    return heatmapBig;
}


float compareHeatmaps(const std::vector<float>& hm1, const std::vector<float>& hm2, float threshold = 0.1) {
    if (hm1.size() != hm2.size()) {
        throw std::runtime_error("Error: Size mismatch between computed and actual heatmap!");
    }
    float mismatchCount = 0, computedNonZeroCount = 0, trueNonZeroCount = 0, trueNegCount = 0, computedNegCount = 0;
    for (size_t i = 0; i < hm1.size(); ++i) {
        if (hm1[i] != 0.0) trueNonZeroCount++;
        if (hm2[i] != 0.0) computedNonZeroCount++;
        if (std::abs(hm1[i] - hm2[i]) > threshold) {
            mismatchCount++;
            //if (mismatchCount <= 10)
             //   std::cout << "Mismatch at index " << i << ": computed = " << computedHeatmap[i] << ", actual = " << heatmap[i] << std::endl;
        }
        if (hm1[i] < 0) trueNegCount++;
        if (hm2[i] < 0) computedNegCount++;
    }
    float matchCount = hm1.size() - mismatchCount;
    float matchRate = matchCount / hm1.size();
//     if (mismatchCount == 0) {
//         std::cout << "Success! The computed heatmap matches the actual heatmap." << std::endl;
//     } else if (matchRate >= 0.4) {
//         std::cout << "The computed heatmap does not match the actual heatmap. Number of matched elements: " << matchCount << " (" << matchRate * 100 << "%)" << std::endl;
//         std::cout << "Non-zero elements in true heatmap: " << trueNonZeroCount << " (" << trueNonZeroCount / hm1.size() * 100 << " %)" << std::endl;
//         std::cout << "Non-zero elements in computed heatmap: " << computedNonZeroCount << " (" << computedNonZeroCount / hm1.size() * 100 << " %)" << std::endl;
//         std::cout << "Negative elements in true heatmap: " << trueNegCount << " (" << trueNegCount / hm1.size() * 100 << " %)" << std::endl;
//         std::cout << "Negative elements in computed heatmap: " << computedNegCount << " (" << computedNegCount / hm1.size() * 100 << " %)" << std::endl;
//     }
    return matchRate;
}


int main(int argc, char** argv) {
    int decimalAccuracy = 2;
    float threshold = 1 / std::pow(10, decimalAccuracy);

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <coloradar_dir> <run_name>" << std::endl;
        return 1;
    }
    fs::path coloradarDir = argv[1];
    std::string runName = argv[2];

    coloradar::ColoradarDataset dataset(coloradarDir);
    coloradar::ColoradarRun run = dataset.getRun(runName);
//     std::vector<float> heatmap = run.getHeatmap(0, &dataset.cascadeConfig);
//     std::cout << "Read heatmap of size " << heatmap.size() << std::endl;

//     std::vector<int> perm = {0, 1, 2, 3, 4};  // Initial order of dimensions
//     // Loop through all 120 permutations
//     int permutationCount = 1;
//     do {
//         std::vector<int16_t> datacube = run.getDatacube(0, &dataset.cascadeConfig);
//         std::vector<int16_t> rearrangedCube = rearrangeArray5D(datacube, perm);
//         std::vector<float> computedHeatmap = coloradar::cubeToHeatmap(rearrangedCube, &dataset.cascadeConfig);
//
//         float mismatchCount = 0, computedNonZeroCount = 0, trueNonZeroCount = 0;
//         for (size_t j = 0; j < heatmap.size(); ++j) {
//             if (heatmap[j] != 0.0) trueNonZeroCount++;
//             if (computedHeatmap[j] != 0.0) computedNonZeroCount++;
//             if (std::abs(computedHeatmap[j] - heatmap[j]) > threshold) {
//                 mismatchCount++;
//             }
//         }
//         std::cout << "Permutation " << permutationCount << " processed." << std::endl;
//         if (mismatchCount == 0) {
//             std::cout << "Success! The computed heatmap matches the actual heatmap." << std::endl;
//         } else {
//             std::cout << "The computed heatmap does not match the actual heatmap. Number of matched elements: " << heatmap.size() - mismatchCount << " (" << (heatmap.size() - mismatchCount) / heatmap.size() * 100 << "%)" << std::endl;
//             std::cout << "Non-zero elements in true heatmap: " << trueNonZeroCount << " (" << trueNonZeroCount / heatmap.size() * 100 << " %)" << std::endl;
//             std::cout << "Non-zero elements in computed heatmap: " << computedNonZeroCount << " (" << computedNonZeroCount / heatmap.size() * 100 << " %)" << std::endl;
//         }
//         std::cout << std::endl;
//
//         permutationCount++;
//     } while (std::next_permutation(perm.begin(), perm.end()) && permutationCount <= 120);

    for (size_t i = 0; i < 10; ++i) {
        std::vector<float> heatmap = run.getHeatmap(i, &dataset.cascadeConfig);
        for (size_t j = 0; j < 10; ++j) {
            std::vector<int16_t> datacube = run.getDatacube(j, &dataset.cascadeConfig);
            std::vector<float> computedHeatmap = coloradar::cubeToHeatmap(datacube, &dataset.cascadeConfig);
//             for (int p = 0; p <= 23; ++p) {
//                 std::vector<float> rearrangedComputedHeatmap = rearrangeArray(computedHeatmap, i);
//                 float matchRate = compareHeatmaps(heatmap, rearrangedComputedHeatmap, threshold);
//                 if (matchRate >= 0.5) {
//                     std::cout << "Rate " << matchRate << " calculated for cube number " << j << " and heatmap number " << i << " (computed heatmap perm number " << p << ")" << std::endl << std::endl;
//                     // std::cout << "Rate " << matchRate << " calculated for cube number " << j << " and heatmap number " << i << std::endl << std::endl;
//                     // std::cout << "WARNING"<< std::endl;
//                 }
//             }
            float matchRate = compareHeatmaps(heatmap, computedHeatmap, threshold);
            if (matchRate >= 0.5) {
                std::cout << "Rate " << matchRate << " calculated for cube number " << j << " and heatmap number " << i << std::endl << std::endl;
            }
        }
    }
//     std::vector<int16_t> datacube = run.getDatacube(3, &dataset.cascadeConfig);
//     std::cout << "Read cube of size " << datacube.size() << std::endl;
//
//     std::vector<float> computedHeatmap = coloradar::cubeToHeatmap(datacube, &dataset.cascadeConfig);
//     std::cout << "Computed heatmap of size " << computedHeatmap.size() << std::endl;

//     for (int i = 0; i <= 23; ++i) {
//         std::vector<float> rearrangedArray = rearrangeArray(computedHeatmap, i);
//         float mismatchCount = 0, computedNonZeroCount = 0, trueNonZeroCount = 0;
//         for (size_t j = 0; j < heatmap.size(); ++j) {
//             if (heatmap[j] != 0.0) trueNonZeroCount++;
//             if (rearrangedArray[j] != 0.0) computedNonZeroCount++;
//             if (std::abs(rearrangedArray[j] - heatmap[j]) > threshold) {
//                 mismatchCount++;
//             }
//         }
//         std::cout << "Permutation " << i << " processed." << std::endl;
//         if (mismatchCount == 0) {
//             std::cout << "Success! The computed heatmap matches the actual heatmap." << std::endl;
//         } else {
//             std::cout << "The computed heatmap does not match the actual heatmap. Number of matched elements: " << heatmap.size() - mismatchCount << " (" << (heatmap.size() - mismatchCount) / heatmap.size() * 100 << "%)" << std::endl;
//             std::cout << "Non-zero elements in true heatmap: " << trueNonZeroCount << " (" << trueNonZeroCount / heatmap.size() * 100 << " %)" << std::endl;
//             std::cout << "Non-zero elements in computed heatmap: " << computedNonZeroCount << " (" << computedNonZeroCount / heatmap.size() * 100 << " %)" << std::endl;
//         }
//         std::cout << std::endl;
//     }

//     bool match = true;
//     if (computedHeatmap.size() != heatmap.size()) {
//         std::cerr << "Error: Size mismatch between computed and actual heatmap!" << std::endl;
//         return 1;
//     }
//     float mismatchCount = 0, computedNonZeroCount = 0, trueNonZeroCount = 0, trueNegCount = 0, computedNegCount = 0;
//     for (size_t i = 0; i < computedHeatmap.size(); ++i) {
//         if (heatmap[i] != 0.0) trueNonZeroCount++;
//         if (computedHeatmap[i] != 0.0) computedNonZeroCount++;
//         if (std::abs(computedHeatmap[i] - heatmap[i]) > threshold) {
//             mismatchCount++;
//             //if (mismatchCount <= 10)
//              //   std::cout << "Mismatch at index " << i << ": computed = " << computedHeatmap[i] << ", actual = " << heatmap[i] << std::endl;
//         }
//         if (heatmap[i] < 0) trueNegCount++;
//         if (computedHeatmap[i] < 0) computedNegCount++;
//     }
//     if (match && mismatchCount == 0) {
//         std::cout << "Success! The computed heatmap matches the actual heatmap." << std::endl;
//     } else {
//         std::cout << "The computed heatmap does not match the actual heatmap. Number of matched elements: " << computedHeatmap.size() - mismatchCount << " (" << (computedHeatmap.size() - mismatchCount) / computedHeatmap.size() * 100 << "%)" << std::endl;
//         std::cout << "Non-zero elements in true heatmap: " << trueNonZeroCount << " (" << trueNonZeroCount / heatmap.size() * 100 << " %)" << std::endl;
//         std::cout << "Non-zero elements in computed heatmap: " << computedNonZeroCount << " (" << computedNonZeroCount / computedHeatmap.size() * 100 << " %)" << std::endl;
//         std::cout << "Negative elements in true heatmap: " << trueNegCount << " (" << trueNegCount / heatmap.size() * 100 << " %)" << std::endl;
//         std::cout << "Negative elements in computed heatmap: " << computedNegCount << " (" << computedNegCount / computedHeatmap.size() * 100 << " %)" << std::endl;
//     }
    // compareVectorsIgnoringOrder(heatmap, computedHeatmap);

    return 0;
}
