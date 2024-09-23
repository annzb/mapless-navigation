#include "coloradar_tools.h"
#include <unordered_map>


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
        std::cerr << "Usage: " << argv[0] << " <coloradarDir> [<runName>] [verticalFov=<degrees>] [horizontalFov=<degrees>] [range=<meters>] [applyTransform=<none>|cascade|single_chip]" << std::endl;
        return -1;
    }
    double verticalFov = args.find("verticalFov") != args.end() ? std::stod(args["verticalFov"]) : 180.0;
    double horizontalFov = args.find("horizontalFov") != args.end() ? std::stod(args["horizontalFov"]) : 360.0;
    double range = args.find("range") != args.end() ? std::stod(args["range"]) : 0.0;
    std::string applyTransform = (args.find("applyTransform") != args.end()) ? args["applyTransform"] : "";
    if (!applyTransform.empty() && applyTransform != "cascade" && applyTransform != "single_chip") {
        std::cerr << "Error: Invalid applyTransform option '" << applyTransform << "'. Valid options are 'cascade', 'single_chip', or unspecified." << std::endl;
        return -1;
    }

    coloradar::ColoradarDataset dataset(coloradarDir);
    std::vector<std::string> targetRuns;
    if (!runName.empty()) {
        targetRuns.push_back(runName);
    }

    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    if (applyTransform == "cascade") {
        transform = dataset.getBaseToCascadeRadarTransform().inverse();
    } else if (applyTransform == "single_chip") {
        transform = dataset.getBaseToRadarTransform().inverse();
    }

    for (size_t i = 0; i < targetRuns.size(); ++i) {
        coloradar::ColoradarRun run = dataset.getRun(targetRuns[i]);
        auto poses = run.getPoses<octomath::Pose6D>();
        std::vector<double> gtTimestamps = run.getPoseTimestamps();
        if (applyTransform == "cascade") {
            std::vector<double> cascadeTimestamps = run.getCascadeTimestamps();
            poses = run.interpolatePoses(poses, gtTimestamps, cascadeTimestamps);
        } else if (applyTransform == "single_chip") {
            std::vector<double> radarTimestamps = run.getRadarTimestamps();
            poses = run.interpolatePoses(poses, gtTimestamps, radarTimestamps);
        }
        run.sampleMapFrames(horizontalFov, verticalFov, range, transform, poses);
    }

//
//    fs::path coloradarDirPath(coloradarDir);
//    std::string sampleRun = targetRuns[0];
//    coloradar::ColoradarRun run = dataset.getRun(sampleRun);
//    pcl::PointCloud<pcl::PointXYZI> map = run.readLidarOctomap();
//    pcl::PointCloud<pcl::PointXYZI> baseToCascadeMap;
//    pcl::PointCloud<pcl::PointXYZI> cascadeToBaseMap;
//    pcl::transformPointCloud(map, baseToCascadeMap, cascadeTransform);
//    pcl::transformPointCloud(map, cascadeToBaseMap, cascadeTransform.inverse());
//    fs::path outputDir = coloradarDirPath / "test_output";
//    pcl::io::savePCDFile(outputDir / (sampleRun + "_base_to_cascade.pcd"), baseToCascadeMap);
//    pcl::io::savePCDFile(outputDir / (sampleRun + "_cascade_to_base.pcd"), cascadeToBaseMap);

    return 0;
}