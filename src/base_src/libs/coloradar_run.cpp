#include "utils.h"
#include "coloradar_tools.h"

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <fstream>
#include <sstream>
#include <stdexcept>


template<typename PoseT>
PoseT readPose(std::istringstream* iss);

template<>
octomath::Pose6D readPose<octomath::Pose6D>(std::istringstream* iss) {
    octomath::Vector3 translation;
    octomath::Quaternion rotation;
    *iss >> translation.x() >> translation.y() >> translation.z() >> rotation.x() >> rotation.y() >> rotation.z() >> rotation.u();
    return octomath::Pose6D(translation, rotation);
}
template<>
Eigen::Affine3f readPose<Eigen::Affine3f>(std::istringstream* iss) {
    Eigen::Vector3f translation;
    Eigen::Quaternionf rotation;
    *iss >> translation.x() >> translation.y() >> translation.z() >> rotation.x() >> rotation.y() >> rotation.z() >> rotation.w();
    Eigen::Affine3f pose;
    pose.translate(translation);
    pose.rotate(rotation);
    return pose;
}


coloradar::ColoradarRun::ColoradarRun(const std::filesystem::path& runPath) : runDirPath(runPath), name(runDirPath.filename()) {
    checkPathExists(runDirPath);
    posesDirPath = runDirPath / "groundtruth";
    checkPathExists(posesDirPath);
    lidarScansDirPath = runDirPath / "lidar";
    checkPathExists(lidarScansDirPath);
    radarScansDirPath = runDirPath / "single_chip";
    checkPathExists(radarScansDirPath);
    pointcloudsDirPath = lidarScansDirPath / "pointclouds";
    checkPathExists(pointcloudsDirPath);
    lidarMapsDirPath = runDirPath / "lidar_maps";
}

std::vector<double> coloradar::ColoradarRun::getPoseTimestamps() {
    std::filesystem::path tsFilePath = posesDirPath / "timestamps.txt";
    return readTimestamps(tsFilePath);
}

std::vector<double> coloradar::ColoradarRun::getLidarTimestamps() {
    std::filesystem::path tsFilePath = lidarScansDirPath / "timestamps.txt";
    return readTimestamps(tsFilePath);
}

std::vector<double> coloradar::ColoradarRun::getRadarTimestamps() {
    std::filesystem::path tsFilePath = radarScansDirPath / "timestamps.txt";
    return readTimestamps(tsFilePath);
}

template<typename PoseT>
std::vector<PoseT> coloradar::ColoradarRun::getPoses() {
    std::filesystem::path posesFilePath = posesDirPath / "groundtruth_poses.txt";
    checkPathExists(posesFilePath);

    std::vector<PoseT> poses;
    std::ifstream infile(posesFilePath);
    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        PoseT pose = readPose<PoseT>(&iss);
        poses.push_back(pose);
    }
    return poses;
}

template<typename CloudT, typename PointT>
CloudT coloradar::ColoradarRun::getLidarPointCloud(const std::filesystem::path& binPath) {
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
    std::filesystem::path pclBinFilePath = pointcloudsDirPath / ("lidar_pointcloud_" + std::to_string(cloudIdx) + ".bin");
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
    std::vector<octomath::Pose6D> poses = getPoses<octomath::Pose6D>();
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

void coloradar::ColoradarRun::saveLidarOctomap(const octomap::OcTree& tree) {
    pcl::PointCloud<pcl::PointXYZI> treePcl = coloradar::octreeToPcl(tree);
    createDirectoryIfNotExists(lidarMapsDirPath);
    std::filesystem::path outputMapFile = lidarMapsDirPath / "map.pcd";
    pcl::io::savePCDFile(outputMapFile, treePcl);
}

pcl::PointCloud<pcl::PointXYZI> coloradar::ColoradarRun::readLidarOctomap() {
    pcl::PointCloud<pcl::PointXYZI> cloud;
    std::filesystem::path mapFilePath = lidarMapsDirPath / "map.pcd";
    checkPathExists(mapFilePath);
    pcl::io::loadPCDFile<pcl::PointXYZI>(mapFilePath.string(), cloud);
    return cloud;
}

void coloradar::ColoradarRun::sampleMapFrames(const float& horizontalFov, const float& verticalFov, const float& range) {
    std::vector<Eigen::Affine3f> poses = getPoses<Eigen::Affine3f>();
    pcl::PointCloud<pcl::PointXYZI> mapCloud = readLidarOctomap();

    for (size_t i = 0; i < poses.size(); ++i) {
        pcl::PointCloud<pcl::PointXYZI> centeredCloud;
        pcl::transformPointCloud(mapCloud, centeredCloud, poses[i]);
        // coloradar::filterFov(pcl::PointCloud<pcl::PointXYZ>& cloud, const float& horizontalFov, const float& verticalFov, const float& range);
    }
}

std::vector<double> coloradar::ColoradarRun::readTimestamps(const std::filesystem::path& path) {
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
