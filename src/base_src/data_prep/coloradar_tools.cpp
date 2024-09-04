#include "coloradar_tools.h"

#include <pcl/common/transforms.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>


void checkPathExists(const fs::path& path) {
    if (!fs::exists(path)) {
        throw std::runtime_error("Directory or file not found: " + path.string());
    }
}


pcl::PointCloud<pcl::PointXYZ> filterFov(pcl::PointCloud<pcl::PointXYZ> cloud, float horizontalFov, float verticalFov, float range) {
    if (horizontalFov == 360 && verticalFov == 360 && range == 0) {
        return cloud;
    }
    Eigen::Matrix3Xf points = cloud.getMatrixXfMap(3, 4, 0);
    Eigen::Array<bool, Eigen::Dynamic, 1> filter = Eigen::Array<bool, Eigen::Dynamic, 1>::Constant(points.cols(), true);
    pcl::PointCloud<pcl::PointXYZ> filtered_cloud;

    if (horizontalFov < 360) {
        float tan_azimuth = std::tan(M_PI / 2 - horizontalFov / 2 * M_PI / 360);
        Eigen::Array<bool, Eigen::Dynamic, 1> azimuth_mask = (points.row(1).array().square() >= (tan_azimuth * points.row(0).array()).square());
        filter = filter && azimuth_mask;
    }
    if (verticalFov < 360) {
        float tan_elevation = std::tan(M_PI / 2 - verticalFov / 2 * M_PI / 360);
        Eigen::Array<bool, Eigen::Dynamic, 1> elevation_mask = (points.row(1).array().square() >= (tan_elevation * points.row(2).array()).square());
        filter = filter && elevation_mask;
    }
    if (range > 0) {
        Eigen::Array<bool, Eigen::Dynamic, 1> range_mask = (points.row(0).array().square() + points.row(1).array().square() + points.row(2).array().square()).sqrt().array() <= range;
        filter = filter && range_mask;
    }
    for (int i = 0; i < cloud.size(); ++i) {
        if (filter(i)) {
            filtered_cloud.push_back(cloud.points[i]);
        }
    }
    return filtered_cloud;
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

//template<typename T>
//std::vector<T> ColoradarRun::getPoses() {
//    fs::path posesFilePath = posesDirPath / "groundtruth_poses.txt";
//    checkPathExists(posesFilePath);
//
//    std::vector<T> poses;
//    std::ifstream infile(posesFilePath);
//    std::string line;
//
//    while (std::getline(infile, line)) {
//        std::istringstream iss(line);
//        Eigen::Vector3f translation;
//        Eigen::Quaternionf rotation;
//        iss >> translation.x() >> translation.y() >> translation.z() >> rotation.x() >> rotation.y() >> rotation.z() >> rotation.w();
//        Eigen::Affine3f pose = Eigen::Affine3f::Identity();
//        pose.translate(translation);
//        pose.rotate(rotation);
//        poses.push_back(pose);
//    }
//    return poses;
//}

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

template<typename CloudT, typename PointT>
CloudT ColoradarRun::getLidarPointCloud(const fs::path& binPath) {
    checkPathExists(binPath);
    std::ifstream infile(binPath, std::ios::binary);
    if (!infile) {
        throw std::runtime_error("Failed to open file: " + binPath.string());
    }
    infile.seekg(0, std::ios::end);
    size_t numPoints = infile.tellg() / (4 * sizeof(float));
    infile.seekg(0, std::ios::beg);

    CloudT cloud;
    cloud.reserve(numPoints);

    for (size_t i = 0; i < numPoints; ++i) {
        float x, y, z;
        infile.read(reinterpret_cast<char*>(&x), sizeof(float));
        infile.read(reinterpret_cast<char*>(&y), sizeof(float));
        infile.read(reinterpret_cast<char*>(&z), sizeof(float));
        infile.ignore(sizeof(float)); // Skip the intensity value
        cloud.push_back(PointT(x, y, z));
    }
    if (cloud.size() < 1) {
        throw std::runtime_error("Failed to read or empty point cloud: " + binPath.string());
    }
    return cloud;
}

template<typename CloudT, typename PointT>
CloudT ColoradarRun::getLidarPointCloud(int cloudIdx) {
    fs::path pclBinFilePath = pointcloudsDirPath / ("lidar_pointcloud_" + std::to_string(cloudIdx) + ".bin");
    return getLidarPointCloud<CloudT, PointT>(pclBinFilePath);
}

octomap::OcTree ColoradarRun::buildLidarOctomap(
    const double& mapResolution,
    const float& lidarTotalHorizontalFov,
    const float& lidarTotalVerticalFov,
    const float& lidarMaxRange,
    Eigen::Affine3f lidarTransform
) {
    if (lidarTotalHorizontalFov <= 0 || lidarTotalHorizontalFov > 360) {
        throw std::runtime_error("Invalid horizontal FOV value: expected 0 < H <= 360, got " + std::to_string(lidarTotalHorizontalFov));
    }
    if (lidarTotalVerticalFov <= 0 || lidarTotalVerticalFov > 360) {
        throw std::runtime_error("Invalid vertical FOV value: expected 0 < V <= 360, got " + std::to_string(lidarTotalVerticalFov));
    }
    if (lidarMaxRange < 0) {
        throw std::runtime_error("Invalid max range value: expected R >= 0 (0 for no filter), got " + std::to_string(lidarMaxRange));
    }

    std::vector<double> lidarTimestamps = getLidarTimestamps();
    std::vector<double> poseTimestamps = getPoseTimestamps();
    std::vector<Eigen::Affine3f> poses = getPoses();
    octomap::OcTree tree(mapResolution);

    for (size_t i = 0; i < lidarTimestamps.size(); ++i) {
        // pcl::PointCloud<pcl::PointXYZ> cloud = getLidarPointCloud<pcl::PointCloud<pcl::PointXYZ>, pcl::PointXYZ>(i);
        octomap::Pointcloud cloud = getOctoLidarPointCloud(i);

        double lidarTimestamp = lidarTimestamps[i];
        int poseIdx = findClosestEarlierTimestamp(lidarTimestamp, poseTimestamps);
        Eigen::Affine3f pose = poses[poseIdx];
        // octomap::pose6d octoPose(octomap::point3d(pose.translation().x(), pose.translation().y(), pose.translation().z()), octomap::)

        // pcl::PointCloud<pcl::PointXYZ> filteredCloud = filterFov(cloud, lidarTotalHorizontalFov, lidarTotalVerticalFov, lidarMaxRange);
        // pcl::transformPointCloud(filteredCloud, filteredCloud, lidarTransform);
        // pcl::transformPointCloud(filteredCloud, filteredCloud, pose);

//        octomap::Pointcloud octomapCloud;
//        for (const auto& point : filteredCloud.points) {
//            octomapCloud.push_back(point.x, point.y, point.z);
//        }
//        std::cout << i;
//        tree.insertPointCloud(octomapCloud, octomap::point3d(pose.translation().x(), pose.translation().y(), pose.translation().z()));
//        std::cout << " done" << std::endl;
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
