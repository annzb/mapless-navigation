#include <rclcpp/rclcpp.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/frustum_culling.h>
#include <octomap/octomap.h>
#include <octomap/OcTree.h>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Transform.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <filesystem>
#include <vector>
#include <limits>
#include <cmath>
#include <unordered_map>


namespace fs = std::filesystem;

struct Pose {
    double x, y, z, qx, qy, qz, qw;
};

std::vector<Pose> readGroundTruthPoses(const std::string& filepath) {
    std::vector<Pose> poses;
    std::ifstream infile(filepath);
    std::string line;
    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        Pose pose;
        iss >> pose.x >> pose.y >> pose.z >> pose.qx >> pose.qy >> pose.qz >> pose.qw;
        poses.push_back(pose);
    }
    return poses;
}

std::vector<double> readTimestamps(const std::string& filepath) {
    std::vector<double> timestamps;
    std::ifstream infile(filepath);
    std::string line;
    while (std::getline(infile, line)) {
        timestamps.push_back(std::stod(line));
    }
    return timestamps;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr readPointCloudFromBin(const std::string& filepath) {
    std::ifstream infile(filepath, std::ios::binary);
    if (!infile) {
        std::cerr << "Failed to open file: " << filepath << std::endl;
        return nullptr;
    }
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointXYZ point;
    while (infile.read(reinterpret_cast<char*>(&point.x), sizeof(float)) &&
           infile.read(reinterpret_cast<char*>(&point.y), sizeof(float)) &&
           infile.read(reinterpret_cast<char*>(&point.z), sizeof(float))) {
        cloud->points.push_back(point);
    }
    return cloud;
}

void filterPointCloudWithFOV(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float horizontal_fov, float vertical_fov, float range) {
    pcl::FrustumCulling<pcl::PointXYZ> fc;
    fc.setInputCloud(cloud);
    fc.setVerticalFOV(vertical_fov);     // vertical FOV in degrees
    fc.setHorizontalFOV(horizontal_fov); // horizontal FOV in degrees
    fc.setNearPlaneDistance(0.0);        // near plane distance (fixed at 0 for now)
    fc.setFarPlaneDistance(range);       // maximum range
    Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity(); // You can set this based on your camera setup
    fc.setCameraPose(camera_pose);
    pcl::PointCloud<pcl::PointXYZ> filtered_cloud;
    fc.filter(filtered_cloud);
    cloud->swap(filtered_cloud);  // Replace original cloud with the filtered one
}

void applyTransformations(octomap::OcTree& tree, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, const Pose& pose) {
    // Static transform: lidar to sensor frame
    Eigen::Affine3f static_transform = Eigen::Affine3f::Identity();
    static_transform.translation() << 0.0, 0.0, 0.03618;
    static_transform.rotate(Eigen::AngleAxisf(M_PI, Eigen::Vector3f::UnitZ()));
    pcl::transformPointCloud(*cloud, *cloud, static_transform);

    // Dynamic transform: sensor frame to global frame
    Eigen::Affine3f dynamic_transform = Eigen::Affine3f::Identity();
    dynamic_transform.translation() << pose.x, pose.y, pose.z;
    Eigen::Quaternionf quat(pose.qw, pose.qx, pose.qy, pose.qz);
    dynamic_transform.rotate(quat);
    pcl::transformPointCloud(*cloud, *cloud, dynamic_transform);
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

int main(int argc, char** argv) {
    auto args = parseArguments(argc, argv);
    std::string input_data_path;
    if (args.find("input_data_path") != args.end()) {
        input_data_path = args["input_data_path"];
    } else if (argc > 1) {
        input_data_path = argv[1];
    } else {
        std::cerr << "Usage: " << argv[0] << " <input_data_path> [vertical_fov=<degrees>] [horizontal_fov=<degrees>] [range=<meters>]" << std::endl;
        return -1;
    }

    double vertical_fov = args.find("vertical_fov") != args.end() ? std::stod(args["vertical_fov"]) : 30.0;   // Default: 60 degrees total (30 up, 30 down)
    double horizontal_fov = args.find("horizontal_fov") != args.end() ? std::stod(args["horizontal_fov"]) : 60.0; // Default: 120 degrees total (60 left, 60 right)
    double range = args.find("range") != args.end() ? std::stod(args["range"]) : 10.0;                         // Default: 10 meters
    if (horizontal_fov <= 0 || horizontal_fov > 180 || vertical_fov <= 0 || vertical_fov > 180 || range < 0) {
        std::cerr << "FOV values must be between 0 and 180 degrees, range must be positive" << std::endl;
        return -1;
    }

    // Read lidar timestamps and ground truth poses
    std::string lidar_timestamps_path = input_data_path + "/lidar/timestamps.txt";
    std::string groundtruth_timestamps_path = input_data_path + "/groundtruth/timestamps.txt";
    std::string groundtruth_poses_path = input_data_path + "/groundtruth/groundtruth_poses.txt";

    std::vector<double> lidar_timestamps = readTimestamps(lidar_timestamps_path);
    std::vector<double> groundtruth_timestamps = readTimestamps(groundtruth_timestamps_path);
    std::vector<Pose> groundtruth_poses = readGroundTruthPoses(groundtruth_poses_path);

    // Initialize Octomap tree
    octomap::OcTree tree(0.25);  // 0.25m resolution

    // Process point clouds one by one to avoid memory overflow
    for (size_t i = 0; /*i < lidar_timestamps.size()*/ i<500; ++i) {
        std::string lidar_bin_file = input_data_path + "/lidar/pointclouds/lidar_pointcloud_" + std::to_string(i) + ".bin";
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = readPointCloudFromBin(lidar_bin_file);

        if (!cloud || cloud->empty()) {
            std::cerr << "Failed to read point cloud: " << lidar_bin_file << std::endl;
            continue;
        }

        double lidar_timestamp = lidar_timestamps[i];
        Pose closest_pose = groundtruth_poses[i];

        std::cout << "Transforming and filtering cloud " << lidar_bin_file << " with " << cloud->size() << " points." << std::endl;

        filterPointCloudWithFOV(cloud, horizontal_fov, vertical_fov, range);
        applyTransformations(tree, cloud, closest_pose);

        // Insert filtered point cloud into Octomap
        for (const auto& point : cloud->points) {
            tree.updateNode(octomap::point3d(point.x, point.y, point.z), true);
        }

        std::cout << "Filtered and inserted cloud with " << cloud->size() << " points into Octomap" << std::endl;

        cloud->clear(); // Clear the cloud explicitly
        cloud.reset();
    }

    std::cout << "Total number of nodes in the octomap: " << tree.size() << std::endl;
    std::cout << "Total number of leaf nodes (occupied space) in the octomap: " << tree.getNumLeafNodes() << std::endl;

    // Save the Octomap to a .bt file
    std::string output_file = std::string(getenv("HOME")) + "/test_output/" + fs::path(input_data_path).stem().string() + "_octomap.bt";
    std::cout << output_file << std::endl;
    tree.writeBinary(output_file);
    std::cout << "Octomap saved to " << output_file << std::endl;

    return 0;
}
