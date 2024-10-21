#include <iostream>
#include "coloradar_tools.h"
#include <set>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <iomanip>


namespace fs = std::filesystem;


int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <coloradar_dir> <run_name>" << std::endl;
        return 1;
    }
    fs::path coloradarDir = argv[1];
    std::string runName = argv[2];

    fs::path coloradarDirPath(coloradarDir);
    coloradar::ColoradarDataset dataset(coloradarDirPath);
    coloradar::ColoradarRun run = dataset.getRun(runName);
    run.createRadarPointclouds(&dataset.cascadeConfig);

    return 0;
}
