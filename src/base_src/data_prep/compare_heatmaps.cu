#include <iostream>
#include "coloradar_cuda.h"
#include <set>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>


namespace fs = std::filesystem;


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
            // std::cout << "Mismatch at index " << i << ": actual = " << hm1[i] << ", computed = " << hm2[i] << std::endl;
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
    bool compareUnordered = false
) {
    float threshold = 1 / std::pow(10, decimalAccuracy);
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
    // printArrays(referenceHeatmap, computedHeatmap, 10);
}

void compareHeatmapArrays(
    std::vector<std::vector<float>> referenceHeatmaps, std::vector<std::vector<float>> computedHeatmaps,
    int decimalAccuracy, float matchRateThreshold = 0.0,
    std::string referenceDescription = "", std::string computedDescription = "",
    bool compareUnordered = false
) {
    for (size_t referenceIdx = 0; referenceIdx < referenceHeatmaps.size(); ++referenceIdx) {
        for (size_t computedIdx = 0; computedIdx < computedHeatmaps.size(); ++computedIdx) {
            report(
                referenceHeatmaps[referenceIdx], computedHeatmaps[computedIdx],
                decimalAccuracy, matchRateThreshold,
                referenceDescription + " " + std::to_string(referenceIdx), computedDescription + " " + std::to_string(computedIdx),
                compareUnordered
            );
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

std::vector<std::vector<float>> collectHeatmaps(
    std::vector<int> idx,
    coloradar::ColoradarDataset dataset,
    coloradar::ColoradarRun run,
    coloradar::RadarProcessor* radarProcessor,
    bool fromCubes = false,
    bool collapseDoppler = false,
    bool removeAntennaCoupling = false,
    bool applyPhaseFrequencyCalib = false
) {
    std::vector<std::vector<float>> heatmaps;
    std::vector<float> hm;
    for (auto const& i : idx) {
        if (fromCubes){
            auto cube = run.getDatacube(i, &dataset.cascadeConfig);
            hm = radarProcessor->cubeToHeatmap(cube, collapseDoppler, removeAntennaCoupling, applyPhaseFrequencyCalib);
//             std::cout << "Cube " << i << std::endl;
//             for (size_t j = 0; j < 20; ++j) {
//                 std::cout << hm[j] << std::endl;
//             }
        } else {
            hm = run.getHeatmap(i, &dataset.cascadeConfig);
        }
        heatmaps.push_back(hm);
    }
    return heatmaps;
}


int main(int argc, char** argv) {
    int decimalAccuracy = 4;
    float matchRateThreshold = 0.7;
    bool compareUnordered = false;
    bool collapseDoppler = true, removeAntennaCoupling = true, applyPhaseFrequencyCalib = true;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <coloradar_dir> <run_name>" << std::endl;
        return 1;
    }
    fs::path coloradarDir = argv[1];
    std::string runName = argv[2];

    fs::path coloradarDirPath(coloradarDir);
    coloradar::ColoradarDataset dataset(coloradarDirPath);
    coloradar::ColoradarRun run = dataset.getRun(runName);
    coloradar::RadarProcessor radarProcessor(&dataset.cascadeConfig);

    fs::path dHmFolder = coloradarDirPath / "heatmaps2";
    fs::path containerHmFolder = coloradarDirPath / "ros_output" / "heatmaps521_bins";
    // fs::path aCubesFolder = coloradarDirPath / "ros_output" / "single_cube_bins";
    std::vector<std::string> dHmFilenames = collectHeatmapFilenames(dHmFolder);  // = {"unsorted_heatmap2_0.bin"};
    // std::vector<std::string> containerHmFilenames = {"heatmap_0.bin", "heatmap_1.bin", "heatmap_2.bin", "heatmap_3.bin", "heatmap_4.bin", "heatmap_5.bin", "heatmap_6.bin", "heatmap_7.bin", "heatmap_8.bin", "heatmap_9.bin"};
    // std::vector<std::string> containerHmFilenames = {"heatmap_3.bin"};
    std::vector<std::string> containerHmFilenames = collectHeatmapFilenames(containerHmFolder);
    std::vector<int> datasetHmIdx(50); std::iota(std::begin(datasetHmIdx), std::end(datasetHmIdx), 0); // = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, };
    std::vector<int> computedHmIdx(523); std::iota(std::begin(computedHmIdx), std::end(computedHmIdx), 0);
    // std::vector<int> computedHmIdx = {15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
    // std::vector<int> computedHmIdx = {5};

    std::string dDescription = "doncey's";
    std::string containerDescription = "container";
    std::string datasetDescription = "dataset";
    std::string computedDescription = "computed";
    std::vector<std::vector<float>> dHms = collectHeatmaps(dHmFolder, dHmFilenames, dataset, run);
    std::vector<std::vector<float>> containerHms = collectHeatmaps(containerHmFolder, containerHmFilenames, dataset, run);
    std::vector<std::vector<float>> datasetHms = collectHeatmaps(datasetHmIdx, dataset, run, &radarProcessor);
    std::vector<std::vector<float>> computedHms = collectHeatmaps(computedHmIdx, dataset, run, &radarProcessor, true, collapseDoppler, removeAntennaCoupling, applyPhaseFrequencyCalib);

    // compareHeatmapArrays(datasetHms, datasetHms, decimalAccuracy, matchRateThreshold, datasetDescription, datasetDescription, compareUnordered, permuteComputed, findBestArrangement);
    // compareHeatmapArrays(datasetHms, computedHms, decimalAccuracy, matchRateThreshold, datasetDescription, computedDescription, compareUnordered, permuteComputed, findBestArrangement);
    // compareHeatmapArrays(datasetHms, dHms, decimalAccuracy, matchRateThreshold, datasetDescription, dDescription, compareUnordered, permuteComputed, findBestArrangement);
    // compareHeatmapArrays(datasetHms, containerHms, decimalAccuracy, matchRateThreshold, datasetDescription, containerDescription, compareUnordered, permuteComputed, findBestArrangement);
    // compareHeatmapArrays(dHms, computedHms, decimalAccuracy, matchRateThreshold, dDescription, computedDescription, compareUnordered, permuteComputed, findBestArrangement);
    // compareHeatmapArrays(dHms, containerHms, decimalAccuracy, matchRateThreshold, dDescription, containerDescription, compareUnordered, permuteComputed, findBestArrangement);
    compareHeatmapArrays(computedHms, containerHms, decimalAccuracy, matchRateThreshold, computedDescription, containerDescription, compareUnordered);

    return 0;
}
