#ifndef COLORADAR_TOOLS_H
#define COLORADAR_TOOLS_H

#include <Eigen/Dense>
#include <octomap/octomap.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <filesystem>
#include <string>
#include <vector>


namespace fs = std::filesystem;


class ColoradarRun {
public:
    ColoradarRun(const fs::path& runPath);

    std::vector<double> getPoseTimestamps();
    std::vector<double> getLidarTimestamps();
    std::vector<double> getRadarTimestamps();
    std::vector<Eigen::Affine3f> getPoses();

    pcl::PointCloud<pcl::PointXYZ>::Ptr getLidarPointCloud(const fs::path& binPath);
    pcl::PointCloud<pcl::PointXYZ>::Ptr getLidarPointCloud(int cloudIdx);

    octomap::OcTree buildLidarOctomap(
        const double& mapResolution,
        const float& lidarTotalHorizontalFov,
        const float& lidarTotalVerticalFov,
        const float& lidarMaxRange,
        Eigen::Affine3f lidarTransform = Eigen::Affine3f::Identity()
    );

private:
    fs::path runDirPath;
    fs::path posesDirPath;
    fs::path lidarScansDirPath;
    fs::path radarScansDirPath;
    fs::path pointcloudsDirPath;

    std::vector<double> readTimestamps(const fs::path& path);
    int findClosestEarlierTimestamp(const double& targetTs, const std::vector<double>& timestamps);
    void filterFov(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float horizontalFov, float verticalFov, float range);
    void filterRange(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float range)
};


class ColoradarDataset {
public:
    ColoradarDataset(const fs::path& coloradarPath);

    Eigen::Affine3f getBaseToLidarTransform();
    Eigen::Affine3f getBaseToRadarTransform();
    std::vector<std::string> listRuns();
    ColoradarRun getRun(const std::string& runName);

private:
    fs::path coloradarDirPath;
    fs::path calibDirPath;
    fs::path transformsDirPath;
    fs::path runsDirPath;

    Eigen::Affine3f loadTransform(const fs::path& filePath);
};

#endif
