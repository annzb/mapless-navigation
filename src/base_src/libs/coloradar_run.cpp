#include "coloradar_tools.h"

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <fstream>
#include <sstream>


coloradar::ColoradarRun::ColoradarRun(const std::filesystem::path& runPath) : runDirPath(runPath), name(runDirPath.filename()) {
    coloradar::internal::checkPathExists(runDirPath);
    posesDirPath = runDirPath / "groundtruth";
    coloradar::internal::checkPathExists(posesDirPath);
    lidarScansDirPath = runDirPath / "lidar";
    coloradar::internal::checkPathExists(lidarScansDirPath);
    radarScansDirPath = runDirPath / "single_chip";
    coloradar::internal::checkPathExists(radarScansDirPath);
    cascadeScansDirPath = runDirPath / "cascade";
    coloradar::internal::checkPathExists(cascadeScansDirPath);
    cascadeHeatmapsDirPath = cascadeScansDirPath / "heatmaps";
    coloradar::internal::checkPathExists(cascadeHeatmapsDirPath);
    lidarCloudsDirPath = lidarScansDirPath / "pointclouds";
    coloradar::internal::checkPathExists(lidarCloudsDirPath);
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

std::vector<double> coloradar::ColoradarRun::getCascadeTimestamps() {
    std::filesystem::path tsFilePath = cascadeHeatmapsDirPath / "timestamps.txt";
    return readTimestamps(tsFilePath);
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
        OctoPointcloud cloud = getLidarPointCloud<coloradar::OctoPointcloud>(i);
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
    pcl::PointCloud<pcl::PointXYZI> treePcl;
    coloradar::octreeToPcl(tree, treePcl);
    coloradar::internal::createDirectoryIfNotExists(lidarMapsDirPath);
    std::filesystem::path outputMapFile = lidarMapsDirPath / "map.pcd";
    pcl::io::savePCDFile(outputMapFile, treePcl);
}

pcl::PointCloud<pcl::PointXYZI> coloradar::ColoradarRun::readLidarOctomap() {
    pcl::PointCloud<pcl::PointXYZI> cloud;
    std::filesystem::path mapFilePath = lidarMapsDirPath / "map.pcd";
    coloradar::internal::checkPathExists(mapFilePath);
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
    coloradar::internal::checkPathExists(path);
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
