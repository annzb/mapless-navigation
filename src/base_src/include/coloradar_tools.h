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


namespace coloradar {

void filterFov(pcl::PointCloud<pcl::PointXYZ>& cloud, const float& horizontalFov, const float& verticalFov, const float& range);


class OctoPointcloud : public octomap::Pointcloud {
public:
    OctoPointcloud() = default;
    OctoPointcloud(const OctoPointcloud& other) : octomap::Pointcloud(other) {}
    OctoPointcloud(const pcl::PointCloud<pcl::PointXYZ>& cloud);

    pcl::PointCloud<pcl::PointXYZ> toPcl();

    void filterFov(const float& horizontalFovTan, const float& verticalFovTan, const float& range);
    void transform(const Eigen::Affine3f& transformMatrix);
    using octomap::Pointcloud::transform;
};


class ColoradarRun {
public:
    ColoradarRun(const fs::path& runPath);

    std::vector<double> getPoseTimestamps();
    std::vector<double> getLidarTimestamps();
    std::vector<double> getRadarTimestamps();
    std::vector<octomath::Pose6D> getPoses();

    pcl::PointCloud<pcl::PointXYZ> getPclLidarPointCloud(const fs::path& binPath) { return getLidarPointCloud<pcl::PointCloud<pcl::PointXYZ>, pcl::PointXYZ>(binPath); }
    pcl::PointCloud<pcl::PointXYZ> getPclLidarPointCloud(int cloudIdx) { return getLidarPointCloud<pcl::PointCloud<pcl::PointXYZ>, pcl::PointXYZ>(cloudIdx); }
    OctoPointcloud getOctoLidarPointCloud(const fs::path& binPath) { return getLidarPointCloud<OctoPointcloud, octomap::point3d>(binPath); }
    OctoPointcloud getOctoLidarPointCloud(int cloudIdx) { return getLidarPointCloud<OctoPointcloud, octomap::point3d>(cloudIdx); }

    octomap::OcTree buildLidarOctomap(
        const double& mapResolution,
        const float& lidarTotalHorizontalFov,
        const float& lidarTotalVerticalFov,
        const float& lidarMaxRange,
        Eigen::Affine3f lidarTransform = Eigen::Affine3f::Identity()
    );

protected:
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

protected:
    fs::path coloradarDirPath;
    fs::path calibDirPath;
    fs::path transformsDirPath;
    fs::path runsDirPath;

    Eigen::Affine3f loadTransform(const fs::path& filePath);
};

}

#endif
