#include "coloradar_tools.h"

#include <stdexcept>


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


void savePoses(const std::vector<octomath::Pose6D>& poses, const fs::path& path) {
    std::ofstream outfile(path);
    for (size_t i = 0; i < poses.size(); ++i) {
        const octomath::Vector3& translation = poses[i].trans();
        const octomath::Quaternion& quat = poses[i].rot();
        outfile << translation.x() << " " << translation.y() << " " << translation.z();
        outfile << " " << quat.x() << " " << quat.y() << " " << quat.z() << " " << quat.u() << std::endl;
    }
    outfile.close();
}


int main(int argc, char** argv) {
    auto args = parseArguments(argc, argv);
    std::string coloradarDir = (args.find("coloradarDir") != args.end()) ? args["coloradarDir"] : (argc > 1 ? argv[1] : "");
    std::string runName = (args.find("runName") != args.end()) ? args["runName"] : (argc > 2 ? argv[2] : "");
    if (coloradarDir.empty() || runName.empty()) {
        std::cerr << "Usage: " << argv[0] << "<coloradarDir> <runName> [outputFilePath=<str>]" << std::endl;
        return -1;
    }
    std::string outputFile = args.find("outputFilePath") != args.end() ? args["outputFilePath"] : "";
    fs::path outputFilePath(outputFile.empty() ? "lidar_poses_interpolated.txt" : outputFile);

    coloradar::ColoradarDataset dataset(coloradarDir);
    coloradar::ColoradarRun run = dataset.getRun(runName);
    std::vector<double> lidarTimestamps = run.getLidarTimestamps();
    std::vector<double> poseTimestamps = run.getPoseTimestamps();
    auto poses = run.getPoses<octomath::Pose6D>();
    auto posesInterpolated = run.interpolatePoses(poses, poseTimestamps, lidarTimestamps);

    savePoses(posesInterpolated, outputFilePath);
}
