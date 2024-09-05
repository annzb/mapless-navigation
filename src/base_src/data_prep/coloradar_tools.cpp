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


pcl::PointCloud<pcl::PointXYZ> coloradar::filterFov(pcl::PointCloud<pcl::PointXYZ> cloud, float horizontalFovTan, float verticalFovTan, float range) {
    Eigen::Matrix3Xf points = cloud.getMatrixXfMap(3, 4, 0);
    Eigen::Array<bool, Eigen::Dynamic, 1> azimuthMask = (points.row(1).array().square() >= (horizontalFovTan * points.row(0).array()).square());
    Eigen::Array<bool, Eigen::Dynamic, 1> elevationMask = (points.row(1).array().square() >= (verticalFovTan * points.row(2).array()).square());
    Eigen::Array<bool, Eigen::Dynamic, 1> rangeMask = (points.row(0).array().square() + points.row(1).array().square() + points.row(2).array().square()).sqrt().array() <= range;
    Eigen::Array<bool, Eigen::Dynamic, 1> filter = azimuthMask && elevationMask && rangeMask;

    pcl::PointCloud<pcl::PointXYZ> filteredCloud;
    for (int i = 0; i < cloud.size(); ++i) {
        if (filter(i)) {
            filteredCloud.push_back(cloud.points[i]);
        }
    }
    return filteredCloud;
}


// OctoPointcloud class
coloradar::OctoPointcloud::OctoPointcloud(const pcl::PointCloud<pcl::PointXYZ>& cloud) {
    this->clear();
    this->reserve(cloud.size());
    for (const auto& point : cloud.points) {
        this->push_back(octomap::point3d(point.x, point.y, point.z));
    }
}

pcl::PointCloud<pcl::PointXYZ> coloradar::OctoPointcloud::toPcl() {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.reserve(this->size());
    for (size_t i = 0; i < this->size(); ++i) {
        const octomap::point3d& point = this->getPoint(i);
        cloud.push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
    }
    return cloud;
}

void coloradar::OctoPointcloud::transform(const Eigen::Affine3f& transformMatrix) {
    Eigen::Quaternionf rotation(transformMatrix.rotation());
    octomath::Pose6D transformPose(
        octomath::Vector3(transformMatrix.translation().x(), transformMatrix.translation().y(), transformMatrix.translation().z()),
        octomath::Quaternion(rotation.w(), rotation.x(), rotation.y(), rotation.z())
    );
    this->transform(transformPose);
}

bool coloradar::OctoPointcloud::checkAzimuthFrontOnly(const octomap::point3d& point, float horizontalFovTan) {
    return (point.y() >= point.x() / horizontalFovTan && point.y() >= -point.x() / horizontalFovTan);
}
bool coloradar::OctoPointcloud::checkAzimuthFrontBack(const octomap::point3d& point, float horizontalFovTan) {
    return (point.y() >= point.x() / horizontalFovTan || point.y() >= -point.x() / horizontalFovTan);
}
bool coloradar::OctoPointcloud::checkElevationFrontOnly(const octomap::point3d& point, float verticalFovTan) {
    return (point.y() >= point.z() / verticalFovTan && point.y() >= -point.z() / verticalFovTan);
}
bool coloradar::OctoPointcloud::checkElevationFrontBack(const octomap::point3d& point, float verticalFovTan) {
    return (point.y() >= point.z() / verticalFovTan || point.y() >= -point.z() / verticalFovTan);
}

void coloradar::OctoPointcloud::filterFov(const float& horizontalFov, const float& verticalFov, const float& range) {
    if (horizontalFov <= 0 || horizontalFov > 360) {
        throw std::runtime_error("Invalid horizontal FOV value: expected 0 < FOV <= 360, got " + std::to_string(horizontalFov));
    }
    if (verticalFov <= 0 || verticalFov > 360) {
        throw std::runtime_error("Invalid vertical FOV value: expected 0 < FOV <= 360, got " + std::to_string(verticalFov));
    }
    if (range <= 0) {
        throw std::runtime_error("Invalid max range value: expected R > 0, got " + std::to_string(range));
    }

    coloradar::OctoPointcloud unfilteredCloud(*this);
    this->clear();

    float horizontalFovTan = std::tan(horizontalFov / 2 * M_PI / 360.0f);
    float verticalFovTan = std::tan(verticalFov / 2 * M_PI / 360.0f);
    FovCheck checkAzimuth = horizontalFov <= 180 ? checkAzimuthFrontOnly : checkAzimuthFrontBack;
    FovCheck checkElevation = verticalFov <= 180 ? checkElevationFrontOnly : checkElevationFrontBack;

    for (size_t i = 0; i < unfilteredCloud.size(); ++i) {
        const octomap::point3d& point = unfilteredCloud.getPoint(i);
        float distance = point.norm();
        if (distance > range) {
            continue;
        }
        if (checkAzimuth(point, horizontalFovTan) && checkElevation(point, verticalFovTan)) {
            this->push_back(point);
        }
    }
}


// GPT VERSION

//bool coloradar::OctoPointcloud::checkAzimuthFrontOnly(const octomap::point3d& point, float horizontalFovTan) {
//    return (std::abs(point.y()) <= std::abs(point.x()) * horizontalFovTan);
//}
//
//bool coloradar::OctoPointcloud::checkAzimuthFrontBack(const octomap::point3d& point, float horizontalFovTan) {
//    return true;  // No limitation for 360° FOV
//}
//
//bool coloradar::OctoPointcloud::checkElevationFrontOnly(const octomap::point3d& point, float verticalFovTan) {
//    return (std::abs(point.y()) <= std::abs(point.z()) * verticalFovTan);
//}
//
//bool coloradar::OctoPointcloud::checkElevationFrontBack(const octomap::point3d& point, float verticalFovTan) {
//    return true;  // No limitation for 360° FOV
//}
//
//void coloradar::OctoPointcloud::filterFov(const float& horizontalFov, const float& verticalFov, const float& range) {
//    if (horizontalFov <= 0 || horizontalFov > 360) {
//        throw std::runtime_error("Invalid horizontal FOV value: expected 0 < FOV <= 360, got " + std::to_string(horizontalFov));
//    }
//    if (verticalFov <= 0 || verticalFov > 360) {
//        throw std::runtime_error("Invalid vertical FOV value: expected 0 < FOV <= 360, got " + std::to_string(verticalFov));
//    }
//    if (range <= 0) {
//        throw std::runtime_error("Invalid max range value: expected R > 0, got " + std::to_string(range));
//    }
//
//    coloradar::OctoPointcloud unfilteredCloud(*this);
//    this->clear();
//
//    // Correct tangent calculation (dividing by 180 to convert degrees to radians)
//    float horizontalFovTan = std::tan(horizontalFov / 2 * M_PI / 180.0f);
//    float verticalFovTan = std::tan(verticalFov / 2 * M_PI / 180.0f);
//
//    // Determine which function to use for azimuth and elevation checks
//    FovCheck checkAzimuth = (horizontalFov < 360) ? checkAzimuthFrontOnly : checkAzimuthFrontBack;
//    FovCheck checkElevation = (verticalFov < 360) ? checkElevationFrontOnly : checkElevationFrontBack;
//
//    for (size_t i = 0; i < unfilteredCloud.size(); ++i) {
//        const octomap::point3d& point = unfilteredCloud.getPoint(i);
//        float distance = point.norm();
//        if (distance > range) {
//            continue;  // Skip points outside the maximum range
//        }
//
//        // Perform azimuth and elevation checks
//        if (checkAzimuth(point, horizontalFovTan) && checkElevation(point, verticalFovTan)) {
//            this->push_back(point);  // Point is inside the FOV
//        }
//    }
//}



// ColoradarRun class

coloradar::ColoradarRun::ColoradarRun(const fs::path& runPath) : runDirPath(runPath) {
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

std::vector<double> coloradar::ColoradarRun::getPoseTimestamps() {
    fs::path tsFilePath = posesDirPath / "timestamps.txt";
    return readTimestamps(tsFilePath);
}

std::vector<double> coloradar::ColoradarRun::getLidarTimestamps() {
    fs::path tsFilePath = lidarScansDirPath / "timestamps.txt";
    return readTimestamps(tsFilePath);
}

std::vector<double> coloradar::ColoradarRun::getRadarTimestamps() {
    fs::path tsFilePath = radarScansDirPath / "timestamps.txt";
    return readTimestamps(tsFilePath);
}

std::vector<octomath::Pose6D> coloradar::ColoradarRun::getPoses() {
    fs::path posesFilePath = posesDirPath / "groundtruth_poses.txt";
    checkPathExists(posesFilePath);

    std::vector<octomath::Pose6D> poses;
    std::ifstream infile(posesFilePath);
    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        octomath::Vector3 translation;
        octomath::Quaternion rotation;
        iss >> translation.x() >> translation.y() >> translation.z() >> rotation.x() >> rotation.y() >> rotation.z() >> rotation.u();
        octomath::Pose6D pose(translation, rotation);
        poses.push_back(pose);
    }
    return poses;
}

template<typename CloudT, typename PointT>
CloudT coloradar::ColoradarRun::getLidarPointCloud(const fs::path& binPath) {
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
CloudT coloradar::ColoradarRun::getLidarPointCloud(int cloudIdx) {
    fs::path pclBinFilePath = pointcloudsDirPath / ("lidar_pointcloud_" + std::to_string(cloudIdx) + ".bin");
    return getLidarPointCloud<CloudT, PointT>(pclBinFilePath);
}

octomap::OcTree coloradar::ColoradarRun::buildLidarOctomap(
    const double& mapResolution,
    const float& lidarTotalHorizontalFov,
    const float& lidarTotalVerticalFov,
    const float& lidarMaxRange,
    Eigen::Affine3f lidarTransform
) {
    float maxRange = lidarMaxRange == 0 ? std::numeric_limits<float>::max() : lidarMaxRange;
    std::vector<double> lidarTimestamps = getLidarTimestamps();
    std::vector<double> poseTimestamps = getPoseTimestamps();
    std::vector<octomath::Pose6D> poses = getPoses();
    octomap::OcTree tree(mapResolution);

    for (size_t i = 0; i < lidarTimestamps.size(); ++i) {
        OctoPointcloud cloud = getOctoLidarPointCloud(i);
        double lidarTimestamp = lidarTimestamps[i];
        int poseIdx = findClosestEarlierTimestamp(lidarTimestamp, poseTimestamps);
        octomath::Pose6D pose = poses[poseIdx];

        cloud.filterFov(lidarTotalHorizontalFov, lidarTotalVerticalFov, maxRange);
        cloud.transform(lidarTransform);
        cloud.transform(pose);
        tree.insertPointCloud(cloud, pose.trans());
    }
    return tree;
}

std::vector<double> coloradar::ColoradarRun::readTimestamps(const fs::path& path) {
    checkPathExists(path);
    std::vector<double> timestamps;
    std::ifstream infile(path);
    std::string line;
    while (std::getline(infile, line)) {
        timestamps.push_back(std::stod(line));
    }
    return timestamps;
}

int coloradar::ColoradarRun::findClosestEarlierTimestamp(const double& targetTs, const std::vector<double>& timestamps) {
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

coloradar::ColoradarDataset::ColoradarDataset(const fs::path& coloradarPath) : coloradarDirPath(coloradarPath) {
    checkPathExists(coloradarDirPath);
    calibDirPath = coloradarDirPath / "calib";
    checkPathExists(calibDirPath);
    transformsDirPath = calibDirPath / "transforms";
    checkPathExists(transformsDirPath);
    runsDirPath = coloradarDirPath / "kitti";
    checkPathExists(runsDirPath);
}

Eigen::Affine3f coloradar::ColoradarDataset::getBaseToLidarTransform() {
    fs::path baseToLidarTransformPath = transformsDirPath / "base_to_lidar.txt";
    Eigen::Affine3f transform = loadTransform(baseToLidarTransformPath);
    return transform;
}

Eigen::Affine3f coloradar::ColoradarDataset::getBaseToRadarTransform() {
    fs::path baseToRadarTransformPath = transformsDirPath / "base_to_single_chip.txt";
    Eigen::Affine3f transform = loadTransform(baseToRadarTransformPath);
    return transform;
}

std::vector<std::string> coloradar::ColoradarDataset::listRuns() {
    std::vector<std::string> runs;
    for (const auto& entry : fs::directory_iterator(runsDirPath)) {
        if (entry.is_directory()) {
            runs.push_back(entry.path().filename().string());
        }
    }
    return runs;
}

coloradar::ColoradarRun coloradar::ColoradarDataset::getRun(const std::string& runName) {
    fs::path runPath = runsDirPath / runName;
    return ColoradarRun(runPath);
}

Eigen::Affine3f coloradar::ColoradarDataset::loadTransform(const fs::path& filePath) {
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
