#include <iostream>
#include "coloradar_cuda.h"


namespace fs = std::filesystem;


int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <coloradar_dir> <run_name>" << std::endl;
        return 1;
    }
    fs::path coloradarDir = argv[1];
    std::string runName = argv[2];

    coloradar::ColoradarDataset dataset(coloradarDir);
    coloradar::ColoradarRun run = dataset.getRun(runName);

    std::vector<std::complex<double>> datacube = run.getDatacube(0, &dataset.cascadeConfig);
    std::cout << "Read cube of size " << datacube.size() << std::endl;

    std::vector<float> heatmap = run.getHeatmap(0, &dataset.cascadeConfig);
    std::cout << "Read heatmap of size " << heatmap.size() << std::endl;

    std::vector<float> computedHeatmap = coloradar::cubeToHeatmap(datacube, &dataset.cascadeConfig);
    std::cout << "Computed heatmap of size " << computedHeatmap.size() << std::endl;

    bool match = true;
    float threshold = 1e-3;
    if (computedHeatmap.size() != heatmap.size()) {
        std::cerr << "Error: Size mismatch between computed and actual heatmap!" << std::endl;
        return 1;
    }
    for (size_t i = 0; i < computedHeatmap.size(); ++i) {
        if (std::abs(computedHeatmap[i] - heatmap[i]) > threshold) {
            match = false;
            // std::cout << "Mismatch at index " << i << ": computed = " << computedHeatmap[i] << ", actual = " << heatmap[i] << std::endl;
        }
    }
    if (match) {
        std::cout << "Success! The computed heatmap matches the actual heatmap." << std::endl;
    } else {
        std::cout << "The computed heatmap does not match the actual heatmap." << std::endl;
    }

    return 0;
}
