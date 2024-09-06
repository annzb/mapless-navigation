#include "coloradar_tools.h"
#include "octree_diff.h"

#include <pcl/io/pcd_io.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <stdexcept>
#include <iomanip>


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
        std::cerr << "Usage: " << argv[0] << " <coloradarDir> [<runName>] [mapResolution=<meters>] [verticalFov=<degrees>] [horizontalFov=<degrees>] [range=<meters>]" << std::endl;
        return -1;
    }
    double mapResolution = args.find("mapResolution") != args.end() ? std::stod(args["mapResolution"]) : 0.1;
    double verticalFov = args.find("verticalFov") != args.end() ? std::stod(args["verticalFov"]) : 180.0;
    double horizontalFov = args.find("horizontalFov") != args.end() ? std::stod(args["horizontalFov"]) : 360.0;
    double range = args.find("range") != args.end() ? std::stod(args["range"]) : 0.0;

    fs::path coloradarPath(coloradarDir);
    coloradar::ColoradarDataset dataset(coloradarPath);
    fs::path mapsPath = coloradarPath / "lidar_maps";
    createDirectoryIfNotExists(mapsPath);

    std::vector<std::string> targetRuns;
    if (runName.empty()) {
        targetRuns = dataset.listRuns();
    } else {
        targetRuns.push_back(runName);
    }
    Eigen::Affine3f transform = dataset.getBaseToLidarTransform();

    for (size_t i = 0; i < targetRuns.size(); ++i) {
        coloradar::ColoradarRun run = dataset.getRun(targetRuns[i]);
        octomap::OcTree tree = run.buildLidarOctomap(mapResolution, horizontalFov, verticalFov, range, transform);
        // std::cout << "Built map for run " << targetRuns[i] << ", total number of nodes in the octomap: " << tree.size() << ", number of leaf nodes in the octomap: " << tree.getNumLeafNodes() << std::endl;

        fs::path outputPath = mapsPath / runName;
        createDirectoryIfNotExists(outputPath);
        fs::path outputMapFile = outputPath / "map.pcd";
        pcl::PointCloud<pcl::PointXYZI> treePcl = octreeToPcl(tree);
        pcl::io::savePCDFile(outputMapFile, treePcl);
    }
    // Sample frames for every pose
    //pcl::PointCloud<pcl::PointXYZI>::Ptr treePclPtr = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>(treePcl);
//    for (size_t i = 0; i < groundtruthPoses.size(); ++i) {
//        std::stringstream ss;
//        ss << outputRunDir << "/map_frame_" << i << ".pcd";
//        pcl::PointCloud<pcl::PointXYZI> frame = sampleFrameFromMap(treePclPtr, groundtruthPoses[i], horizontalFov, verticalFov, range);
//        std::cout << "Frame " << i << " size: " << frame.size() << std::endl;
//        pcl::io::savePCDFile(ss.str(), frame);
//    }

//    std::string output_csv_file = outputRunDir + "/total_octomap.csv";
//    saveLeafNodesAsCSV(tree, output_csv_file);

    return 0;
}