#include <iostream>
#include "coloradar_tools.h"
#include <set>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <iomanip>


namespace fs = std::filesystem;


int main(int argc, char** argv) {
    if (argc < 3 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " <coloradar_dir> <run_name> [intensity_threshold_percent]" << std::endl;
        return 1;
    }
    fs::path coloradarDir = argv[1];
    std::string runName = argv[2];
    float intensityThresholdPercent = (argc == 4) ? std::stof(argv[3]) : 0;

    fs::path coloradarDirPath(coloradarDir);
    coloradar::ColoradarDataset dataset(coloradarDirPath);
    coloradar::ColoradarRun run = dataset.getRun(runName);
    run.createRadarPointclouds(&dataset.cascadeConfig, intensityThresholdPercent);

    return 0;
}
