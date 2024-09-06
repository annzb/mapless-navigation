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


void createDirectoryIfNotExists(const fs::path& dirPath) {
    if (!fs::exists(dirPath)) {
        fs::create_directories(dirPath);
    }
}

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

//void saveLeafNodesAsCSV(const octomap::OcTree& tree, const std::string& output_file) {
//    std::ofstream outfile(output_file);
//    if (!outfile) {
//        throw std::runtime_error("Failed to open output file: " + output_file);
//    }
//
//    outfile << "x,y,z,log_odds,probability,occupied\n";
//    for (octomap::OcTree::leaf_iterator it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
//        double x = it.getX();
//        double y = it.getY();
//        double z = it.getZ();
//        float log_odds = it->getLogOdds();
//        float probability = 1.0 / (1.0 + exp(-log_odds)); // Convert log-odds to probability
//        bool occupied = tree.isNodeOccupied(*it);
//        outfile << x << "," << y << "," << z << "," << log_odds << "," << probability << "," << occupied << "\n";
//    }
//    outfile.close();
//    std::cout << "Saved leaf nodes to " << output_file << std::endl;
//}

pcl::PointCloud<pcl::PointXYZI> octreeToPcl(const octomap::OcTree& tree) {
    pcl::PointCloud<pcl::PointXYZI> cloud;
    for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
        octomap::point3d coords = it.getCoordinate();
        pcl::PointXYZI point;
        point.x = coords.x();
        point.y = coords.y();
        point.z = coords.z();
        point.intensity = it->getLogOdds();
        cloud.push_back(point);
    }
    return cloud;
}

void printPointCloud(const pcl::PointCloud<pcl::PointXYZI>& cloud, std::size_t num_points = 5) {
    std::cout << "Point cloud has " << cloud.size() << " points." << std::endl;
    for (std::size_t i = 0; i < std::min(num_points, cloud.size()); ++i) {
        const auto& point = cloud.points[i];
        std::cout << "Point " << i << ": "
                  << "x = " << point.x << ", "
                  << "y = " << point.y << ", "
                  << "z = " << point.z << ", "
                  << "intensity = " << point.intensity << std::endl;
    }
}


int main(int argc, char** argv) {
    auto args = parseArguments(argc, argv);
    //std::cout << argc << " " << args << std::endl;
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
