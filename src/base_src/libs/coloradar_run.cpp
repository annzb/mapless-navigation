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

    radarHeatmapsDirPath = radarScansDirPath / "heatmaps";
    coloradar::internal::checkPathExists(radarHeatmapsDirPath);
    radarCubesDirPath = radarScansDirPath / "adc_samples";
    coloradar::internal::checkPathExists(radarCubesDirPath);
    radarPointcloudsDirPath = radarScansDirPath / "pointclouds";
    coloradar::internal::createDirectoryIfNotExists(radarPointcloudsDirPath);
    coloradar::internal::createDirectoryIfNotExists(radarPointcloudsDirPath / "data");

    cascadeHeatmapsDirPath = cascadeScansDirPath / "heatmaps";
    coloradar::internal::checkPathExists(cascadeHeatmapsDirPath);
    cascadeCubesDirPath = cascadeScansDirPath / "adc_samples";
    coloradar::internal::checkPathExists(cascadeCubesDirPath);
    cascadePointcloudsDirPath = cascadeScansDirPath / "pointclouds";
    coloradar::internal::createDirectoryIfNotExists(cascadePointcloudsDirPath);
    coloradar::internal::createDirectoryIfNotExists(cascadePointcloudsDirPath / "data");

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

std::vector<double> coloradar::ColoradarRun::getCascadeCubeTimestamps() {
    std::filesystem::path tsFilePath = cascadeCubesDirPath / "timestamps.txt";
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

pcl::PointCloud<pcl::PointXYZI> coloradar::ColoradarRun::readMapFrame(const int& frameIdx) {
    pcl::PointCloud<pcl::PointXYZI> cloud;
    std::filesystem::path frameFilePath = lidarMapsDirPath / ("frame_" + std::to_string(frameIdx) + ".pcd");
    coloradar::internal::checkPathExists(frameFilePath);
    pcl::io::loadPCDFile<pcl::PointXYZI>(frameFilePath.string(), cloud);
    return cloud;
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


std::vector<int16_t> coloradar::ColoradarRun::getDatacube(const std::filesystem::path& binFilePath, coloradar::RadarConfig* config) {
    coloradar::internal::checkPathExists(binFilePath);
    std::ifstream file(binFilePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + binFilePath.string());
    }
    int totalElements = config->numTxAntennas * config->numRxAntennas * config->numChirpsPerFrame * config->numAdcSamplesPerChirp * 2;
    std::vector<int16_t> frameBytes(totalElements);
    file.read(reinterpret_cast<char*>(frameBytes.data()), totalElements * sizeof(int16_t));
    if (file.gcount() != totalElements * sizeof(int16_t)) {
        throw std::runtime_error("Datacube file read error or size mismatch");
    }
    file.close();
    return frameBytes;
}

std::vector<int16_t> coloradar::ColoradarRun::getDatacube(const int& cubeIdx, coloradar::RadarConfig* config) {
    std::filesystem::path cubeDirPath;
    if (auto cascadeConfig = dynamic_cast<CascadeConfig*>(config)) {
        cubeDirPath = cascadeCubesDirPath;
    } else {
        cubeDirPath = radarCubesDirPath;
    }
    std::filesystem::path cubeBinFilePath = cubeDirPath / "data" / ("frame_" + std::to_string(cubeIdx) + ".bin");
    return getDatacube(cubeBinFilePath, config);
}

std::vector<float> coloradar::ColoradarRun::getHeatmap(const std::filesystem::path& binFilePath, coloradar::RadarConfig* config) {
    coloradar::internal::checkPathExists(binFilePath);
    std::ifstream file(binFilePath, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + binFilePath.string());
    }
    int totalElements = config->numElevationBins * config->numAzimuthBins * config->numPosRangeBins * 2;
    std::vector<float> heatmap(totalElements);
    file.read(reinterpret_cast<char*>(heatmap.data()), totalElements * sizeof(float));
    if (file.gcount() != totalElements * sizeof(float)) {
        throw std::runtime_error("Heatmap file read error or size mismatch");
    }
    file.close();
    return heatmap;
}

std::vector<float> coloradar::ColoradarRun::getHeatmap(const int& hmIdx, coloradar::RadarConfig* config) {
    std::filesystem::path heatmapDirPath;
    if (auto cascadeConfig = dynamic_cast<CascadeConfig*>(config)) {
        heatmapDirPath = cascadeHeatmapsDirPath;
    } else {
        heatmapDirPath = radarHeatmapsDirPath;
    }
    std::filesystem::path hmBinFilePath = heatmapDirPath / "data" / ("heatmap_" + std::to_string(hmIdx) + ".bin");
    return getHeatmap(hmBinFilePath, config);
}

std::vector<float> coloradar::ColoradarRun::clipHeatmapImage(const std::vector<float>& image, const float& horizontalFov, const float& verticalFov, const float& range, coloradar::RadarConfig* config) {
    if (horizontalFov <= 0 || horizontalFov > 360) {
        throw std::runtime_error("Invalid horizontal FOV value: expected 0 < FOV <= 360, got " + std::to_string(horizontalFov));
    }
    if (verticalFov <= 0 || verticalFov > 180) {
        throw std::runtime_error("Invalid vertical FOV value: expected 0 < FOV <= 180, got " + std::to_string(verticalFov));
    }
    if (range <= 0) {
        throw std::runtime_error("Invalid max range value: expected R > 0, got " + std::to_string(range));
    }
    float horizontalHalfFovRad = horizontalFov / 2 * M_PI / 180.0f;
    float verticalHalfFovRad = verticalFov / 2 * M_PI / 180.0f;

    auto it = std::lower_bound(config->azimuthBins.begin(), config->azimuthBins.end(), -horizontalHalfFovRad);
    int binIdx = std::distance(config->azimuthBins.begin(), --it);
    int azimuthMaxBin = config->numAzimuthBins - binIdx - 1;
    it = std::lower_bound(config->elevationBins.begin(), config->elevationBins.end(), -verticalHalfFovRad);
    binIdx = std::distance(config->elevationBins.begin(), --it);
    int elevationMaxBin = config->numElevationBins - binIdx - 1;
    int rangeMaxBin = static_cast<int>(std::ceil(range / config->rangeBinWidth));
    return clipHeatmapImage(image, azimuthMaxBin, elevationMaxBin, rangeMaxBin, config);
}

std::vector<float> coloradar::ColoradarRun::clipHeatmapImage(const std::vector<float>& image, const int& azimuthMaxBin, const int& elevationMaxBin, const int& rangeMaxBin, coloradar::RadarConfig* config) {
    int azimuthBinLimit = config->numAzimuthBins / 2;
    int elevationBinLimit = config->numElevationBins / 2;
    int rangeBinLimit = config->numRangeBins;
    if (! 0 <= azimuthMaxBin < azimuthBinLimit) {
        throw std::out_of_range("Invalid azimuthMaxBin selected: allowed selection from 0 to " + std::to_string(azimuthBinLimit - 1) + ", got " + std::to_string(azimuthMaxBin));
    }
    if (! 0 <= elevationMaxBin < elevationBinLimit) {
        throw std::out_of_range("Invalid elevationMaxBin selected: allowed selection from 0 to " + std::to_string(elevationBinLimit - 1) + ", got " + std::to_string(elevationMaxBin));
    }
    if (! 0 <= rangeMaxBin < rangeBinLimit) {
        throw std::out_of_range("Invalid rangeMaxBin selected: allowed selection from 0 to " + std::to_string(rangeBinLimit - 1) + ", got " + std::to_string(rangeMaxBin));
    }
    int azimuthLeftBin = azimuthBinLimit - azimuthMaxBin - 1;
    int azimuthRightBin = azimuthBinLimit + azimuthMaxBin;
    int elevationLeftBin = elevationBinLimit - elevationMaxBin - 1;
    int elevationRightBin = elevationBinLimit + elevationMaxBin;
    std::vector<float> clipped;
    for (int e = elevationLeftBin; e <= elevationRightBin; ++e) {
        for (int a = azimuthLeftBin; a <= azimuthRightBin; ++a) {
            for (int r = 0; r <= rangeMaxBin; ++r) {
                for (int n = 0; n < 2; ++n) {
                    int index = (((e * config->numAzimuthBins + a) * config->numRangeBins + r) * 2) + n;
                    clipped.push_back(image[index]);
                }
            }
        }
    }
    return clipped;
}

void coloradar::ColoradarRun::createRadarPointclouds(coloradar::RadarConfig* config, const float& intensityThresholdPercent) {
    std::filesystem::path heatmapDirPath;
    if (auto cascadeConfig = dynamic_cast<CascadeConfig*>(config)) {
        heatmapDirPath = cascadeHeatmapsDirPath;
    } else {
        heatmapDirPath = radarHeatmapsDirPath;
    }
    heatmapDirPath /= "data";
    for (auto const& entry : std::filesystem::directory_iterator(heatmapDirPath)) {
        if (!entry.is_directory() && entry.path().extension() == ".bin") {
            std::filesystem::path heatmapPath = entry.path();
            std::vector<float> heatmap = getHeatmap(heatmapPath, config);
            pcl::PointCloud<coloradar::RadarPoint> cloud = coloradar::heatmapToPointcloud(heatmap, config, intensityThresholdPercent);

            std::filesystem::path cloudPath = cascadePointcloudsDirPath / "data" / coloradar::internal::replaceInFilename(heatmapPath, "heatmap", "radar_pointcloud").filename();
            std::ofstream file(cloudPath, std::ios::out | std::ios::binary);
            if (!file.is_open()) {
                std::cerr << "Unable to open file for writing: " << cloudPath << std::endl;
                return;
            }
            for (size_t i = 0; i < cloud.points.size(); ++i) {
                file.write(reinterpret_cast<const char*>(&cloud.points[i].x), sizeof(float));
                file.write(reinterpret_cast<const char*>(&cloud.points[i].y), sizeof(float));
                file.write(reinterpret_cast<const char*>(&cloud.points[i].z), sizeof(float));
                file.write(reinterpret_cast<const char*>(&cloud.points[i].intensity), sizeof(float));
                file.write(reinterpret_cast<const char*>(&cloud.points[i].doppler), sizeof(float));
            }
            file.close();
        }
    }
}
