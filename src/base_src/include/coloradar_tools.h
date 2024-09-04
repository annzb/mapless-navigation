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


pcl::PointCloud<pcl::PointXYZ> filterFov(pcl::PointCloud<pcl::PointXYZ> cloud, float horizontalFov, float verticalFov, float range);


class ColoradarRun {
public:
    ColoradarRun(const fs::path& runPath);

    std::vector<double> getPoseTimestamps();
    std::vector<double> getLidarTimestamps();
    std::vector<double> getRadarTimestamps();
    std::vector<Eigen::Affine3f> getPoses();

    pcl::PointCloud<pcl::PointXYZ> getPclLidarPointCloud(const fs::path& binPath) { return getLidarPointCloud<pcl::PointCloud<pcl::PointXYZ>, pcl::PointXYZ>(binPath); }
    pcl::PointCloud<pcl::PointXYZ> getPclLidarPointCloud(int cloudIdx) { return getLidarPointCloud<pcl::PointCloud<pcl::PointXYZ>, pcl::PointXYZ>(cloudIdx); }
    octomap::Pointcloud getOctoLidarPointCloud(const fs::path& binPath) { return getLidarPointCloud<octomap::Pointcloud, octomap::point3d>(binPath); }
    octomap::Pointcloud getOctoLidarPointCloud(int cloudIdx) { return getLidarPointCloud<octomap::Pointcloud, octomap::point3d>(cloudIdx); }

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

    template<typename CloudT, typename PointT>
    CloudT getLidarPointCloud(const fs::path& binPath);
    template<typename CloudT, typename PointT>
    CloudT getLidarPointCloud(int cloudIdx);
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
