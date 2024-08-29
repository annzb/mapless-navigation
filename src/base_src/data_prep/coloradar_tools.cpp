#include "coloradar_tools.h"

#include <pcl/common/transforms.h>
#include <pcl/filters/frustum_culling.h>
#include <pcl/filters/crop_sphere.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>


void checkPathExists(const fs::path& path) {
    if (!fs::exists(path)) {
        throw std::runtime_error("Directory or file not found: " + path.string());
    }
}


// ColoradarRun class

ColoradarRun::ColoradarRun(const fs::path& runPath) : runDirPath(runPath) {
    checkPathExists(runDirPath);
    posesDirPath = runDirPath / "groundtruth";
    checkPathExists(posesDirPath);
    lidarScansDirPath = runDirPath / "lidar";
    checkPathExists(lidarScansDirPath);
    radarScansDirPath = runDirPath / "single_chip";
    checkPathExists(radarScansDirPath);
    pointcloudsDirPath = lidarScansDirPath / "pointclouds";
    checkPathExists(pointcloudsDirPath);
}

std::vector<double> ColoradarRun::getPoseTimestamps() {
    fs::path tsFilePath = posesDirPath / "timestamps.txt";
    return readTimestamps(tsFilePath);
}

std::vector<double> ColoradarRun::getLidarTimestamps() {
    fs::path tsFilePath = lidarScansDirPath / "timestamps.txt";
    return readTimestamps(tsFilePath);
}

std::vector<double> ColoradarRun::getRadarTimestamps() {
    fs::path tsFilePath = radarScansDirPath / "timestamps.txt";
    return readTimestamps(tsFilePath);
}

std::vector<Eigen::Affine3f> ColoradarRun::getPoses() {
    fs::path posesFilePath = posesDirPath / "groundtruth_poses.txt";
    checkPathExists(posesFilePath);

    std::vector<Eigen::Affine3f> poses;
    std::ifstream infile(posesFilePath);
    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        Eigen::Vector3f translation;
        Eigen::Quaternionf rotation;
        iss >> translation.x() >> translation.y() >> translation.z() >> rotation.x() >> rotation.y() >> rotation.z() >> rotation.w();
        Eigen::Affine3f pose = Eigen::Affine3f::Identity();
        pose.translate(translation);
        pose.rotate(rotation);
        poses.push_back(pose);
    }
    return poses;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr ColoradarRun::getLidarPointCloud(const fs::path& binPath) {
    checkPathExists(binPath);
    std::ifstream infile(binPath, std::ios::binary);
    if (!infile) {
        std::cerr << "Failed to open file: " << binPath.string() << std::endl;
        return nullptr;
    }
    infile.seekg(0, std::ios::end);
    size_t fileSize = infile.tellg();
    size_t numPoints = fileSize / (4 * sizeof(float));
    infile.seekg(0, std::ios::beg);

    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    cloud->points.resize(numPoints);

    for (size_t i = 0; i < numPoints; ++i) {
        infile.read(reinterpret_cast<char*>(&cloud->points[i].x), sizeof(float));
        infile.read(reinterpret_cast<char*>(&cloud->points[i].y), sizeof(float));
        infile.read(reinterpret_cast<char*>(&cloud->points[i].z), sizeof(float));
        infile.ignore(sizeof(float)); // Skip the intensity value
    }
    if (!cloud || cloud->empty()) {
        throw std::runtime_error("Failed to read or empty point cloud: " + binPath.string());
    }
    return cloud;
}

pcl::PointCloud<pcl::PointXYZ>::Ptr ColoradarRun::getLidarPointCloud(int cloudIdx) {
    fs::path pclBinFilePath = pointcloudsDirPath / ("lidar_pointcloud_" + std::to_string(cloudIdx) + ".bin");
    return getLidarPointCloud(pclBinFilePath);
}

octomap::OcTree ColoradarRun::buildLidarOctomap(
    const double& mapResolution,
    const float& lidarTotalHorizontalFov,
    const float& lidarTotalVerticalFov,
    const float& lidarMaxRange,
    Eigen::Affine3f lidarTransform
) {
    std::vector<double> lidarTimestamps = getLidarTimestamps();
    std::vector<double> poseTimestamps = getPoseTimestamps();
    std::vector<Eigen::Affine3f> poses = getPoses();
    octomap::OcTree tree(mapResolution);

    for (size_t i = 0; i < lidarTimestamps.size(); ++i) {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = getLidarPointCloud(i);

        double lidarTimestamp = lidarTimestamps[i];
        int poseIdx = findClosestEarlierTimestamp(lidarTimestamp, poseTimestamps);
        Eigen::Affine3f pose = poses[poseIdx];

        filterFov(cloud, lidarTotalHorizontalFov, lidarTotalVerticalFov, lidarMaxRange);
        // filterRange(cloud, lidarMaxRange);
        pcl::transformPointCloud(*cloud, *cloud, lidarTransform);
        pcl::transformPointCloud(*cloud, *cloud, pose);

        for (const auto& point : cloud->points) {
            tree.updateNode(octomap::point3d(point.x, point.y, point.z), true);
        }
    }
    return tree;
}

std::vector<double> ColoradarRun::readTimestamps(const fs::path& path) {
    checkPathExists(path);
    std::vector<double> timestamps;
    std::ifstream infile(path);
    std::string line;
    while (std::getline(infile, line)) {
        timestamps.push_back(std::stod(line));
    }
    return timestamps;
}

int ColoradarRun::findClosestEarlierTimestamp(const double& targetTs, const std::vector<double>& timestamps) {
    if (targetTs <= timestamps[0]) {
        return 0;
    }
    if (targetTs >= timestamps.back()) {
        return timestamps.size() - 1;
    }

    int low = 0;
    int high = timestamps.size() - 1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (timestamps[mid] == targetTs) {
            return mid;
        } else if (timestamps[mid] < targetTs) {
            low = mid + 1;
        } else {
            high = mid - 1;
        }
    }
    return high;
}

void ColoradarRun::filterFov(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float horizontalFov, float verticalFov, float range) {
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
    cloud->swap(filtered_cloud);
}

void ColoradarRun::filterRange(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float range) {
    pcl::CropSphere<pcl::PointXYZ> crop_filter;
    crop_filter.setInputCloud(cloud);
    crop_filter.setRadius(range);
    pcl::PointCloud<pcl::PointXYZ> filtered_cloud;
    crop_filter.filter(filtered_cloud);
    cloud->swap(filtered_cloud);
}


// ColoradarDataset class

ColoradarDataset::ColoradarDataset(const fs::path& coloradarPath) : coloradarDirPath(coloradarPath) {
    checkPathExists(coloradarDirPath);
    calibDirPath = coloradarDirPath / "calib";
    checkPathExists(calibDirPath);
    transformsDirPath = calibDirPath / "transforms";
    checkPathExists(transformsDirPath);
    runsDirPath = coloradarDirPath / "kitti";
    checkPathExists(runsDirPath);
}

Eigen::Affine3f ColoradarDataset::getBaseToLidarTransform() {
    fs::path baseToLidarTransformPath = transformsDirPath / "base_to_lidar.txt";
    Eigen::Affine3f transform = loadTransform(baseToLidarTransformPath);
    // std::cout << "base to lidar: translation  " << transform.translation() << std::endl << "rotation" << std::endl << transform.rotation() << std::endl;
    return transform;
}

Eigen::Affine3f ColoradarDataset::getBaseToRadarTransform() {
    fs::path baseToRadarTransformPath = transformsDirPath / "base_to_single_chip.txt";
    Eigen::Affine3f transform = loadTransform(baseToRadarTransformPath);
    // std::cout << "base to radar: translation  " << transform.translation() << std::endl << "rotation" << std::endl << transform.rotation() << std::endl;
    return transform;
}

std::vector<std::string> ColoradarDataset::listRuns() {
    std::vector<std::string> runs;
    for (const auto& entry : fs::directory_iterator(runsDirPath)) {
        if (entry.is_directory()) {
            runs.push_back(entry.path().filename().string());
        }
    }
    return runs;
}

ColoradarRun ColoradarDataset::getRun(const std::string& runName) {
    fs::path runPath = runsDirPath / runName;
    return ColoradarRun(runPath);
}

Eigen::Affine3f ColoradarDataset::loadTransform(const fs::path& filePath) {
    checkPathExists(filePath);
    Eigen::Affine3f transform = Eigen::Affine3f::Identity();
    Eigen::Vector3f translation;
    Eigen::Quaternionf rotation;

    std::ifstream file(filePath);
    std::string line;
    std::getline(file, line);
    std::istringstream iss(line);
    iss >> translation.x() >> translation.y() >> translation.z();

    std::getline(file, line);
    iss.str(line);
    iss.clear();
    iss >> rotation.x() >> rotation.y() >> rotation.z() >> rotation.w();

    transform.translate(translation);
    transform.rotate(rotation);
    return transform;
}
