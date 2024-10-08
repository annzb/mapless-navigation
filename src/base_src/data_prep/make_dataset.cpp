#include "coloradar_tools.h"
#include <unordered_map>


namespace fs = std::filesystem;


std::unordered_map<std::string, std::string> parseArguments(int argc, char** argv) {
    std::unordered_map<std::string, std::string> arguments;
    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);
        if (arg.find("=") != std::string::npos) {
            auto pos = arg.find("=");
            std::string key = arg.substr(0, pos);
            std::string value = arg.substr(pos + 1);
            arguments[key] = value;
        }
    }
    return arguments;
}


int main(int argc, char** argv) {
    auto args = parseArguments(argc, argv);
    std::string coloradarDir = (args.find("coloradarDir") != args.end())
                               ? args["coloradarDir"]
                               : (argc > 1 ? argv[1] : "");
    std::string runName = (args.find("runName") != args.end())
                          ? args["runName"]
                          : (argc > 2 ? argv[2] : "");
    if (coloradarDir.empty()) {
        std::cerr << "Usage: " << argv[0] << " <coloradarDir> [<runName>] [azimuthMaxBin=<idx>] [elevationMaxBin=<idx>] [rangeMaxBin=<idx>]" << std::endl;
        return -1;
    }
    int azimuthMaxBin = args.find("azimuthMaxBin") != args.end() ? std::stod(args["azimuthMaxBin"]) : 0;
    int elevationMaxBin = args.find("elevationMaxBin") != args.end() ? std::stod(args["elevationMaxBin"]) : 0;
    int rangeMaxBin = args.find("rangeMaxBin") != args.end() ? std::stod(args["rangeMaxBin"]) : 0;

    coloradar::ColoradarDataset dataset(coloradarDir);
    std::vector<std::string> targetRuns;
    if (!runName.empty()) {
        targetRuns.push_back(runName);
    }
    // dataset contents
    // 1. lidar pcl frames
    // 2. radar heatmaps clipped
    // 3. true poses and poses interpolated for radar

    for (size_t i = 0; i < targetRuns.size(); ++i) {
        coloradar::ColoradarRun run = dataset.getRun(targetRuns[i]);
        auto poses = run.getPoses<octomath::Pose6D>();
        std::vector<double> poseTimestamps = run.getPoseTimestamps();
        std::vector<double> radarTimestamps = run.getCascadeTimestamps();
        auto radarPoses = run.interpolatePoses(poses, poseTimestamps, radarTimestamps);
        // save true, radar poses
        // save true, radar timestamps

        for (size_t j = 0; j < heatmapTimestamps.size(); ++j) {
            std::vector<float> rawHeatmap = run.getHeatmap(j, &dataset.cascadeConfig);
            std::vector<float> heatmap = run.clipHeatmapImage(rawHeatmap, azimuthMaxBin, elevationMaxBin, rangeMaxBin, &dataset.cascadeConfig);
            // save heatmap
        }
        for (size_t j = 0; j < poseTimestamps.size(); ++j) {
            pcl::PointCloud<pcl::PointXYZI> mapFrame = run.readMapFrame(j);
            // save frame
        }
    }
    return 0;
}
