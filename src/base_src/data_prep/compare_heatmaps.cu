#include <iostream>
#include "coloradar_cuda.h"
#include <set>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
// #include "munkres.h" // Add Munkres implementation
//
// // Function to create a cost matrix based on the absolute differences
// std::vector<std::vector<float>> createCostMatrix(const std::vector<float>& arr1, const std::vector<float>& arr2) {
//     size_t n = arr1.size();
//     std::vector<std::vector<float>> costMatrix(n, std::vector<float>(n));
//
//     for (size_t i = 0; i < n; ++i) {
//         for (size_t j = 0; j < n; ++j) {
//             costMatrix[i][j] = std::fabs(arr1[i] - arr2[j]);
//         }
//     }
//
//     return costMatrix;
// }
//
// // Function to find the best permutation using the Hungarian (Munkres) Algorithm
// std::vector<int> findBestPermutation(const std::vector<float>& arr1, const std::vector<float>& arr2) {
//     size_t n = arr1.size();
//
//     // Create the cost matrix based on absolute differences
//     std::vector<std::vector<float>> costMatrix = createCostMatrix(arr1, arr2);
//
//     // Apply the Munkres (Hungarian) algorithm to find the optimal assignment
//     Munkres munkres;
//     munkres.solve(costMatrix);  // The matrix is modified in place with 0s and 1s
//
//     // Create a vector to store the matching indices
//     std::vector<int> matchingIndices(n, -1);
//
//     // Extract the matching from the modified costMatrix
//     for (size_t i = 0; i < n; ++i) {
//         for (size_t j = 0; j < n; ++j) {
//             if (costMatrix[i][j] == 0) {
//                 matchingIndices[i] = j;
//                 break;
//             }
//         }
//     }
//
//     return matchingIndices;
// }


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

std::vector<int> findBestMatch(const std::vector<float>& arr1, const std::vector<float>& arr2, float threshold) {
    std::vector<int> matchedIndices(arr1.size(), -1);
    std::vector<bool> used(arr2.size(), false);
    for (size_t i = 0; i < arr1.size(); ++i) {
        float bestDifference = std::numeric_limits<float>::max();
        int bestIndex = -1;

        for (size_t j = 0; j < arr2.size(); ++j) {
            if (!used[j]) {
                float difference = std::fabs(arr1[i] - arr2[j]);
                // Check if this difference is smaller than the best so far and within the threshold
                if (difference < bestDifference && difference <= threshold) {
                    bestDifference = difference;
                    bestIndex = j;
                }
            }
        }
        // If a match is found within the threshold, mark it as used and store the index
        if (bestIndex != -1) {
            matchedIndices[i] = bestIndex;
            used[bestIndex] = true;
        }
    }

    return matchedIndices;
}



std::vector<int> compareHeatmaps(const std::vector<float>& hm1, const std::vector<float>& hm2, float threshold = 0.1) {
    float firstComputed = 9203.95; // hm2[0] == 0.0 ? hm2[1] : hm2[0];
    int matchIdx = -1;
    std::vector<int> matchedIdx;
    if (hm1.size() != hm2.size()) {
        throw std::runtime_error("Error: Size mismatch between computed and actual heatmap!");
    }
    float mismatchCount = 0, computedNonZeroCount = 0, trueNonZeroCount = 0, trueNegCount = 0, computedNegCount = 0;
    for (size_t i = 0; i < hm1.size(); ++i) {
        if (matchIdx == -1 && std::abs(hm1[i] - firstComputed) <= threshold) matchIdx = i;
        if (hm1[i] != 0.0) trueNonZeroCount++;
        if (hm2[i] != 0.0) computedNonZeroCount++;
        if (std::abs(hm1[i] - hm2[i]) > threshold) {
            mismatchCount++;
            //if (mismatchCount <= 10)
             //   std::cout << "Mismatch at index " << i << ": computed = " << computedHeatmap[i] << ", actual = " << heatmap[i] << std::endl;
        } else {
            matchedIdx.push_back(i);
            // std::cout << "Match at index " << i << ": actual = " << hm1[i] << ", computed = " << hm2[i] << std::endl;
        }
        if (hm1[i] < 0) trueNegCount++;
        if (hm2[i] < 0) computedNegCount++;
    }
//     if (matchIdx >= 0)
//         std::cout << "Match at index " << matchIdx << ": actual = " << hm1[matchIdx] << ", computed = " << firstComputed << std::endl;
//     else
//         std::cout << "Could not match " << firstComputed << std::endl;
    return matchedIdx;
}


std::vector<float> select(std::vector<int> idx, std::vector<float> arr) {
    if (idx.size() != arr.size()) throw std::runtime_error("Size mismatch");
    std::vector<float> selected(arr.size());
    for (size_t i = 0; i < idx.size(); ++i) {
        selected.push_back(arr[idx[i]]);
    }
    return selected;
}


void printArrays(std::vector<float> arr1, std::vector<float> arr2, int firstNElements = 0) {
    int lim = firstNElements > 0 ? firstNElements : arr1.size();
    if (arr1.size() < lim || arr2.size() < lim) throw std::runtime_error("Size mismatch");
    for (size_t i = 0; i < lim; ++i) {
        std::cout << arr1[i] << " " << arr2[i] << std::endl;
    }
}


void report(
    std::vector<float> referenceHeatmap, std::vector<float> computedHeatmap,
    int decimalAccuracy, float matchRateThreshold = 0.0,
    std::string referenceDescription = "", std::string computedDescription = "",
    bool compareUnordered = false, bool findBestArrangement = false
) {
    float threshold = 1 / std::pow(10, decimalAccuracy);
    if (findBestArrangement) {
//         std::vector<int> bestMatch = findBestPermutation(referenceHeatmap, computedHeatmap, threshold);
//         computedHeatmap = select(bestMatch, computedHeatmap);
    }
    if (compareUnordered) compareVectorsIgnoringOrder(referenceHeatmap, computedHeatmap, decimalAccuracy);
    std::vector<int> matchIdx = compareHeatmaps(referenceHeatmap, computedHeatmap, threshold);

    float matchRate = static_cast<float>(matchIdx.size()) / referenceHeatmap.size();
    if (matchRate >= matchRateThreshold) {
        // std::cout << "Matched idx for computed heatmap " << computedIdx << " and reference heatmap " << referenceIdx << std::endl;
//                     for (size_t idx = 0; idx < matchIdx.size() && idx < 100; ++idx) {
//                         std::cout << idx << " ";
//                     }
        if (!referenceDescription.empty() && !computedDescription.empty()) {
            std::cout << "Comparing " << referenceDescription << " to " << computedDescription << ": ";
        }
        std::cout << "Matched " << matchIdx.size() << " indices from " << matchIdx[0] << " to " << matchIdx[matchIdx.size() - 1] << " (" << matchRate * 100 << "%)" << std::endl << std::endl;
        // std::cout << "Rate " << matchRate << " calculated for cube number " << i << " and sample heatmap " << entry << std::endl << std::endl;
    }
    // printArrays(referenceHeatmap, computedHeatmap, 50);
}


void compareHeatmapArrays(
    std::vector<std::vector<float>> referenceHeatmaps, std::vector<std::vector<float>> computedHeatmaps,
    int decimalAccuracy, float matchRateThreshold = 0.0,
    std::string referenceDescription = "", std::string computedDescription = "",
    bool compareUnordered = false, bool permuteComputed = false, bool findBestArrangement = false
) {
    for (size_t referenceIdx = 0; referenceIdx < referenceHeatmaps.size(); ++referenceIdx) {
        for (size_t computedIdx = 0; computedIdx < computedHeatmaps.size(); ++computedIdx) {
            if (permuteComputed) {
                for (int p = 0; p <= 23; ++p) {
                    std::vector<float> rearrangedHeatmap = rearrangeArray(computedHeatmaps[computedIdx], p);
                    report(
                        referenceHeatmaps[referenceIdx], rearrangedHeatmap,
                        decimalAccuracy, matchRateThreshold,
                        referenceDescription + " " + std::to_string(referenceIdx), computedDescription + " " + std::to_string(computedIdx),
                        compareUnordered, findBestArrangement
                    );
                }
            } else {
                report(
                    referenceHeatmaps[referenceIdx], computedHeatmaps[computedIdx],
                    decimalAccuracy, matchRateThreshold,
                    referenceDescription + " " + std::to_string(referenceIdx), computedDescription + " " + std::to_string(computedIdx),
                    compareUnordered, findBestArrangement
                );
            }
        }
    }
}


std::vector<std::string> collectHeatmapFilenames(fs::path folder) {
    std::vector<std::string> filenames;
    for (auto const& entry : fs::directory_iterator(folder)) {
        if (!entry.is_directory() && entry.path().extension() == ".bin") {
            filenames.push_back(entry.path().filename());
        }
    }
    return filenames;
}

std::vector<float> cubeToHeatmap(std::vector<int16_t> cube) {
    std::vector<float> hm;
    for (const auto& el : cube) hm.push_back(static_cast<float>(el));
    return hm;
}

std::vector<std::vector<float>> collectHeatmaps(fs::path folder, std::vector<std::string> filenames, coloradar::ColoradarDataset dataset, coloradar::ColoradarRun run, bool fromCubes = false) {
    std::vector<std::vector<float>> heatmaps;
    std::vector<float> hm;
    for (auto const& name : filenames) {
        if (fromCubes){
            auto cube = run.getDatacube(folder / name, &dataset.cascadeConfig);
            hm = cubeToHeatmap(cube);
        } else {
            hm = run.getHeatmap(folder / name, &dataset.cascadeConfig);
        }
        heatmaps.push_back(hm);
    }
    return heatmaps;
}

std::vector<std::vector<float>> collectHeatmaps(std::vector<int> idx, coloradar::ColoradarDataset dataset, coloradar::ColoradarRun run, bool fromCubes = false) {
    std::vector<std::vector<float>> heatmaps;
    std::vector<float> hm;
    for (auto const& i : idx) {
        if (fromCubes){
            auto cube = run.getDatacube(i, &dataset.cascadeConfig);
            // hm = cubeToHeatmap(cube);
            hm = coloradar::cubeToHeatmap(cube, &dataset.cascadeConfig);
        } else {
            hm = run.getHeatmap(i, &dataset.cascadeConfig);
        }
        heatmaps.push_back(hm);
    }
    return heatmaps;
}


int main(int argc, char** argv) {
    int decimalAccuracy = 7;
    float matchRateThreshold = 0.05;
    bool compareUnordered = false, permuteComputed = false, findBestArrangement = false;
    // int computedHeatmapsNum = 1;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <coloradar_dir> <run_name>" << std::endl;
        return 1;
    }
    fs::path coloradarDir = argv[1];
    std::string runName = argv[2];

    fs::path coloradarDirPath(coloradarDir);
    fs::path dHmFolder = coloradarDirPath / "heatmaps2";
    fs::path containerHmFolder = coloradarDirPath / "ros_output" / "single_heatmap_loop_bins";
    // fs::path aCubesFolder = coloradarDirPath / "ros_output" / "single_cube_bins";
    std::vector<std::string> dHmFilenames = collectHeatmapFilenames(dHmFolder);  // = {"unsorted_heatmap2_0.bin"};
    std::vector<std::string> containerHmFilenames = {"heatmap_0.bin"};
    std::vector<int> datasetHmIdx = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    std::vector<int> computedHmIdx = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    // std::vector<std::string> aCubesFolderFilenames = {"cube_0.bin"};
    coloradar::ColoradarDataset dataset(coloradarDirPath);
    coloradar::ColoradarRun run = dataset.getRun(runName);

    std::string dDescription = "doncey's";
    std::string containerDescription = "container";
    std::string datasetDescription = "dataset";
    std::string computedDescription = "computed";
    std::vector<std::vector<float>> dHms = collectHeatmaps(dHmFolder, dHmFilenames, dataset, run);
    std::vector<std::vector<float>> containerHms = collectHeatmaps(containerHmFolder, containerHmFilenames, dataset, run);
    std::vector<std::vector<float>> datasetHms = collectHeatmaps(datasetHmIdx, dataset, run);
    std::vector<std::vector<float>> computedHms = collectHeatmaps(computedHmIdx, dataset, run, true);

    // std::vector<std::vector<float>> referenceHeatmaps = collectHeatmaps(aGtFolder, aGtFolderFilenames, dataset, run);
    // std::vector<std::vector<float>> referenceHeatmaps = collectHeatmaps(dGtFolder, dGtFolderFilenames, dataset, run);
    // std::vector<std::vector<float>> referenceHeatmaps = collectHeatmaps({0}, dataset, run, true);

    // std::vector<std::vector<float>> computedHeatmaps = collectHeatmaps(dGtFolder, dGtFolderFilenames, dataset, run);
    // std::vector<std::vector<float>> computedHeatmaps = collectHeatmaps(aCubesFolder, aCubesFolderFilenames, dataset, run, true);

    // std::vector<std::string> referenceHeatmapFilenames = "unsorted_heatmap2_0.bin"; // {"heatmap_0.bin"};
//     for (auto const& entry : fs::directory_iterator(sampleHmFolder)) {
//         if (!entry.is_directory() && entry.path().extension() == ".bin") {
//             referenceHeatmapFilenames.push_back(entry.path().filename());
//         }
//     }

//     std::vector<std::vector<float>> referenceHeatmaps, computedHeatmaps;
//     for (auto const& name : referenceHeatmapFilenames) {
//         referenceHeatmaps.push_back(run.getHeatmap(sampleHmFolder / name, &dataset.cascadeConfig));
//     }
//     std::cout << "Dataset cubes: ";
//     for (size_t i = 0; i < computedHeatmapsNum; ++i) {
//         std::vector<float> hm = run.getHeatmap(i, &dataset.cascadeConfig);
//         computedHeatmaps.push_back(hm);
//         if (i > 0) std::cout << ", ";
//         std::cout << i;
//     }
//     std::cout << std::endl << std::endl;

//     std::cout << "Computed heatmaps: ";
//     for (size_t i = 0; i < computedHeatmapsNum; ++i) {
//         std::vector<int16_t> datacube = run.getDatacube(i, &dataset.cascadeConfig);
//         computedHeatmaps.push_back(coloradar::cubeToHeatmap(datacube, &dataset.cascadeConfig));
//         if (i > 0) std::cout << ", ";
//         std::cout << i;
//     }
//     std::cout << std::endl << std::endl;

    // compareHeatmapArrays(datasetHms, datasetHms, decimalAccuracy, matchRateThreshold, datasetDescription, datasetDescription, compareUnordered, permuteComputed, findBestArrangement);
    // compareHeatmapArrays(datasetHms, computedHms, decimalAccuracy, matchRateThreshold, datasetDescription, computedDescription, compareUnordered, permuteComputed, findBestArrangement);
    // compareHeatmapArrays(datasetHms, dHms, decimalAccuracy, matchRateThreshold, datasetDescription, dDescription, compareUnordered, permuteComputed, findBestArrangement);
    compareHeatmapArrays(datasetHms, containerHms, decimalAccuracy, matchRateThreshold, datasetDescription, containerDescription, compareUnordered, permuteComputed, findBestArrangement);
    // compareHeatmapArrays(dHms, computedHms, decimalAccuracy, matchRateThreshold, dDescription, computedDescription, compareUnordered, permuteComputed, findBestArrangement);
    // compareHeatmapArrays(dHms, containerHms, decimalAccuracy, matchRateThreshold, dDescription, containerDescription, compareUnordered, permuteComputed, findBestArrangement);
    compareHeatmapArrays(computedHms, containerHms, decimalAccuracy, matchRateThreshold, computedDescription, containerDescription, compareUnordered, permuteComputed, findBestArrangement);


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
//
//     for (size_t i = 0; i < 5; ++i) {
//         std::vector<float> heatmap = run.getHeatmap(i, &dataset.cascadeConfig);
//         for (size_t j = 0; j < 5; ++j) {
//             std::vector<int16_t> datacube = run.getDatacube(j, &dataset.cascadeConfig);
//             std::vector<float> computedHeatmap = coloradar::cubeToHeatmap(datacube, &dataset.cascadeConfig);
// //             for (int p = 0; p <= 23; ++p) {
// //                 std::vector<float> rearrangedComputedHeatmap = rearrangeArray(computedHeatmap, i);
// //                 float matchRate = compareHeatmaps(heatmap, rearrangedComputedHeatmap, threshold);
// //                 if (matchRate >= 0.5) {
// //                     std::cout << "Rate " << matchRate << " calculated for cube number " << j << " and heatmap number " << i << " (computed heatmap perm number " << p << ")" << std::endl << std::endl;
// //                     // std::cout << "Rate " << matchRate << " calculated for cube number " << j << " and heatmap number " << i << std::endl << std::endl;
// //                     // std::cout << "WARNING"<< std::endl;
// //                 }
// //             }
//             float matchRate = compareHeatmaps(heatmap, computedHeatmap, threshold);
//             if (matchRate >= 0.3) {
//                 std::cout << "Rate " << matchRate << " calculated for cube number " << j << " and heatmap number " << i << std::endl << std::endl;
//             }
//         }
//     }
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
