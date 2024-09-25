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
    std::vector<octomath::Pose6D> gtPoses = getPoses<octomath::Pose6D>();
    std::vector<octomath::Pose6D> poses = interpolatePoses(gtPoses, poseTimestamps, lidarTimestamps);
    octomap::OcTree tree(mapResolution);

    for (size_t i = 0; i < lidarTimestamps.size(); ++i) {
        OctoPointcloud cloud = getLidarPointCloud<coloradar::OctoPointcloud>(i);
        cloud.filterFov(lidarTotalHorizontalFov, lidarTotalVerticalFov, maxRange);
        cloud.transform(lidarTransform);
        cloud.transform(poses[i]);
        tree.insertPointCloud(cloud, poses[i].trans());
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

void coloradar::ColoradarRun::sampleMapFrames(
    const float& horizontalFov,
    const float& verticalFov,
    const float& range,
    const Eigen::Affine3f& mapPreTransform,
    std::vector<octomath::Pose6D> poses
) {
    float maxRange = range == 0 ? std::numeric_limits<float>::max() : range;
    if (poses.empty())
        poses = getPoses<octomath::Pose6D>();
    pcl::PointCloud<pcl::PointXYZI> origMapCloud = readLidarOctomap();
    pcl::PointCloud<pcl::PointXYZI> mapCloud;
    pcl::transformPointCloud(origMapCloud, mapCloud, mapPreTransform);

    for (size_t i = 0; i < poses.size(); ++i) {
        Eigen::Affine3f pose = coloradar::internal::toEigenPose(poses[i]);
        pcl::PointCloud<pcl::PointXYZI> centeredCloud;
        pcl::transformPointCloud(mapCloud, centeredCloud, pose);
        filterFov(centeredCloud, horizontalFov, verticalFov, maxRange);
        std::filesystem::path frameFilePath = lidarMapsDirPath / ("frame_" + std::to_string(i) + ".pcd");
        pcl::io::savePCDFile(frameFilePath, centeredCloud);
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


//Eigen::Tensor<float, 4> coloradar::ColoradarRun::getHeatmap(const std::filesystem::path& filePath, const int& numElevationBins, const int& numAzimuthBins, const int& numRangeBins) {
//    coloradar::internal::checkPathExists(filePath);
//    std::ifstream file(filePath, std::ios::binary);
//    if (!file.is_open()) {
//        throw std::runtime_error("Failed to open file " + filename);
//    }
//    file.seekg(0, std::ios::end);
//    std::size_t fileSize = file.tellg();
//    file.seekg(0, std::ios::beg);
//    std::vector<char> buffer(fileSize);
//    file.read(buffer.data(), fileSize);
//    file.close();
//
//    std::size_t numFloats = fileSize / 4;
//    std::vector<float> frameVals(numFloats);
//    std::memcpy(frameVals.data(), buffer.data(), fileSize);
//    if (frameVals.size() != numElevationBins * numAzimuthBins * numRangeBins * 2) {
//        throw std::runtime_error("The number of values in the file does not match the expected dimensions.");
//    }
//
//    Eigen::Tensor<float, 4> heatmap(numElevationBins, numAzimuthBins, numRangeBins, 2);
//    std::size_t idx = 0;
//    for (int i = 0; i < numElevationBins; ++i) {
//        for (int j = 0; j < numAzimuthBins; ++j) {
//            for (int k = 0; k < numRangeBins; ++k) {
//                heatmap(i, j, k, 0) = frameVals[idx];
//                heatmap(i, j, k, 1) = frameVals[idx + 1];
//            }
//        }
//    }
//    return heatmap;
//}
