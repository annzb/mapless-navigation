#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/impl/instantiate.hpp>
#include <pcl/point_traits.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/frustum_culling.h>
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <iomanip>

#include "octree_diff.h"


namespace fs = std::filesystem;


void createDirectoryIfNotExists(const std::string& directoryPath) {
    std::filesystem::path dirPath(directoryPath);
    if (!std::filesystem::exists(dirPath)) {
        std::filesystem::create_directories(dirPath);
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

std::vector<Eigen::Affine3f> readGroundTruthPoses(const std::string& filepath) {
    if (!fs::exists(filepath)) {
        throw std::runtime_error("Ground truth poses file not found: " + filepath);
    }
    std::vector<Eigen::Affine3f> poses;
    std::ifstream infile(filepath);
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        Eigen::Vector3f translation;
        Eigen::Quaternionf quat;
        iss >> translation.x() >> translation.y() >> translation.z() >> quat.x() >> quat.y() >> quat.z() >> quat.w();
        Eigen::Affine3f pose = Eigen::Affine3f::Identity();
        pose.translate(translation);
        pose.rotate(quat);
        poses.push_back(pose);
    }
    return poses;
}

std::vector<double> readTimestamps(const std::string& filepath) {
    if (!fs::exists(filepath)) {
        throw std::runtime_error("Timestamps file not found: " + filepath);
    }
    std::vector<double> timestamps;
    std::ifstream infile(filepath);
    std::string line;
    while (std::getline(infile, line)) {
        timestamps.push_back(std::stod(line));
    }
    return timestamps;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr readPointCloudFromBin(const std::string& filepath) {
    if (!fs::exists(filepath)) {
        throw std::runtime_error("Point cloud file not found: " + filepath);
    }
    std::ifstream infile(filepath, std::ios::binary);
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointXYZ point;
    while (infile.read(reinterpret_cast<char*>(&point.x), sizeof(float)) &&
           infile.read(reinterpret_cast<char*>(&point.y), sizeof(float)) &&
           infile.read(reinterpret_cast<char*>(&point.z), sizeof(float))) {
        cloud->points.push_back(point);
    }
    return cloud;
}


void filterFov(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float horizontalFov, float verticalFov, float range) {
    pcl::FrustumCulling<pcl::PointXYZ> fc;
    fc.setInputCloud(cloud);
    fc.setVerticalFOV(verticalFov);
    fc.setHorizontalFOV(horizontalFov);
    fc.setNearPlaneDistance(0.0);
    fc.setFarPlaneDistance(range);
    Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity();
    fc.setCameraPose(camera_pose);
    pcl::PointCloud<pcl::PointXYZ> filtered_cloud;
    fc.filter(filtered_cloud);
    cloud->swap(filtered_cloud);  // Replace original cloud with the filtered one
}

void transformGlobal(octomap::OcTree& tree, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const Eigen::Affine3f& pose) {
    Eigen::Affine3f static_transform = Eigen::Affine3f::Identity();
    static_transform.translation() << 0.0, 0.0, 0.03618;
    static_transform.rotate(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitZ()));
    pcl::transformPointCloud(*cloud, *cloud, static_transform);
    pcl::transformPointCloud(*cloud, *cloud, pose);
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

pcl::PointCloud<pcl::PointXYZI> sampleFrameFromMap(pcl::PointCloud<pcl::PointXYZI>::Ptr map_pcl, const Eigen::Affine3f& pose, double horizontalFov, double verticalFov, double range) {
    pcl::FrustumCulling<pcl::PointXYZI> fc;
    fc.setInputCloud(map_pcl);
    fc.setVerticalFOV(verticalFov);
    fc.setHorizontalFOV(horizontalFov);
    fc.setNearPlaneDistance(0.0);
    fc.setFarPlaneDistance(range);
    fc.setCameraPose(pose.inverse().matrix());
    pcl::PointCloud<pcl::PointXYZI> filtered_cloud;
    fc.filter(filtered_cloud);
    return filtered_cloud;
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
    std::string inputDataPath;
    if (args.find("inputDataPath") != args.end()) {
        inputDataPath = args["inputDataPath"];
    } else if (argc > 1) {
        inputDataPath = argv[1];
    } else {
        std::cerr << "Usage: " << argv[0] << " <inputDataPath> [mapResolution=<meters>] [verticalFov=<degrees>] [horizontalFov=<degrees>] [range=<meters>]" << std::endl;
        return -1;
    }

    std::string outputFolder = std::string(getenv("HOME")) + "/coloradar/lidar_maps";
    createDirectoryIfNotExists(outputFolder);
    std::string outputRunFolder = outputFolder + "/" + fs::path(inputDataPath).stem().string();
    createDirectoryIfNotExists(outputRunFolder);

    double mapResolution = args.find("mapResolution") != args.end() ? std::stod(args["mapResolution"]) : 0.1;
    double verticalFov = args.find("verticalFov") != args.end() ? std::stod(args["verticalFov"]) : 30.0;       // Default: 60 degrees total (30 up, 30 down)
    double horizontalFov = args.find("horizontalFov") != args.end() ? std::stod(args["horizontalFov"]) : 60.0; // Default: 120 degrees total (60 left, 60 right)
    double range = args.find("range") != args.end() ? std::stod(args["range"]) : 10.0;                            // Default: 10 meters
    if (horizontalFov <= 0 || horizontalFov > 180 || verticalFov <= 0 || verticalFov > 180 || range < 0) {
        std::cerr << "FOV values must be between 0 and 180 degrees, range must be positive" << std::endl;
        return -1;
    }

    // std::string lidarTimestampsPath = inputDataPath + "/lidar/timestamps.txt";
    std::string groundtruthPosesPath = inputDataPath + "/groundtruth/groundtruth_poses.txt";

    // std::vector<double> lidarTimestamps = readTimestamps(lidarTimestampsPath);
    std::vector<Eigen::Affine3f> groundtruthPoses = readGroundTruthPoses(groundtruthPosesPath);
    octomap::OcTree tree(0.25);

    // Build map
    for (size_t i = 0; i < groundtruthPoses.size(); ++i) {
        std::string lidarBinFile = inputDataPath + "/lidar/pointclouds/lidar_pointcloud_" + std::to_string(i) + ".bin";
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = readPointCloudFromBin(lidarBinFile);
        if (!cloud || cloud->empty()) {
            throw std::runtime_error("Failed to read or empty point cloud: " + lidarBinFile);
        }

        // double lidarTimestamp = lidarTimestamps[i];
        Eigen::Affine3f pose = groundtruthPoses[i];
        // std::cout << "pose" << pose.translation() << std::endl;
        filterFov(cloud, horizontalFov, verticalFov, range);
        // filterFov(cloud, 90, 45, 50);
        //std::cout << "points after filter" << cloud->size() << std::endl;
        transformGlobal(tree, cloud, pose);
        //std::cout << "points after transform" << cloud->size() << std::endl << std::endl;

        for (const auto& point : cloud->points) {
            tree.updateNode(octomap::point3d(point.x, point.y, point.z), true);
        }
        cloud->clear();
        cloud.reset();
    }
    std::cout << "Total number of nodes in the octomap: " << tree.size() << std::endl;
    std::cout << "Total number of leaf nodes in the octomap: " << tree.getNumLeafNodes() << std::endl;
    
    std::string outputMapFile = outputRunFolder + "/map.pcd";
    pcl::PointCloud<pcl::PointXYZI> treePcl = octreeToPcl(tree);
    std::cout << "Tree point cloud size: " << treePcl.size() << std::endl;
    pcl::io::savePCDFile(outputMapFile, treePcl); 

    // Sample frames for every pose
    //pcl::PointCloud<pcl::PointXYZI>::Ptr treePclPtr = boost::make_shared<pcl::PointCloud<pcl::PointXYZI>>(treePcl);
//    for (size_t i = 0; i < groundtruthPoses.size(); ++i) {
//        std::stringstream ss;
//        ss << outputRunFolder << "/map_frame_" << i << ".pcd";
//        pcl::PointCloud<pcl::PointXYZI> frame = sampleFrameFromMap(treePclPtr, groundtruthPoses[i], horizontalFov, verticalFov, range);
//        std::cout << "Frame " << i << " size: " << frame.size() << std::endl;
//        pcl::io::savePCDFile(ss.str(), frame);
//    }

//    std::string output_csv_file = outputRunFolder + "/total_octomap.csv";
//    saveLeafNodesAsCSV(tree, output_csv_file);

    return 0;
}
