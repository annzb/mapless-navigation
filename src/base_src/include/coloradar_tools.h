#ifndef COLORADAR_TOOLS_H
#define COLORADAR_TOOLS_H

#include <Eigen/Dense>
#include <octomap/octomap.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <filesystem>
#include <string>
#include <vector>


namespace coloradar {

pcl::PointCloud<pcl::PointXYZI> octreeToPcl(const octomap::OcTree& tree);

template<typename PointT>
void filterFov(pcl::PointCloud<PointT>& cloud, const float& horizontalFov, const float& verticalFov, const float& range);


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
    const std::string name;

    ColoradarRun(const std::filesystem::path& runPath);

    std::vector<double> getPoseTimestamps();
    std::vector<double> getLidarTimestamps();
    std::vector<double> getRadarTimestamps();

    template<typename PoseT>
    std::vector<PoseT> getPoses();

    pcl::PointCloud<pcl::PointXYZ> getPclLidarPointCloud(const std::filesystem::path& binPath) { return getLidarPointCloud<pcl::PointCloud<pcl::PointXYZ>, pcl::PointXYZ>(binPath); }
    pcl::PointCloud<pcl::PointXYZ> getPclLidarPointCloud(int cloudIdx) { return getLidarPointCloud<pcl::PointCloud<pcl::PointXYZ>, pcl::PointXYZ>(cloudIdx); }
    OctoPointcloud getOctoLidarPointCloud(const std::filesystem::path& binPath) { return getLidarPointCloud<OctoPointcloud, octomap::point3d>(binPath); }
    OctoPointcloud getOctoLidarPointCloud(int cloudIdx) { return getLidarPointCloud<OctoPointcloud, octomap::point3d>(cloudIdx); }

    octomap::OcTree buildLidarOctomap(
        const double& mapResolution,
        const float& lidarTotalHorizontalFov,
        const float& lidarTotalVerticalFov,
        const float& lidarMaxRange,
        Eigen::Affine3f lidarTransform = Eigen::Affine3f::Identity()
    );
    void saveLidarOctomap(const octomap::OcTree& tree);
    pcl::PointCloud<pcl::PointXYZI> readLidarOctomap();

    void sampleMapFrames(const float& horizontalFov, const float& verticalFov, const float& range);

protected:
    std::filesystem::path runDirPath;
    std::filesystem::path posesDirPath;
    std::filesystem::path lidarScansDirPath;
    std::filesystem::path radarScansDirPath;
    std::filesystem::path pointcloudsDirPath;
    std::filesystem::path lidarMapsDirPath;

    std::vector<double> readTimestamps(const std::filesystem::path& path);
    int findClosestEarlierTimestamp(const double& targetTs, const std::vector<double>& timestamps);

    template<typename CloudT, typename PointT>
    CloudT getLidarPointCloud(const std::filesystem::path& binPath);
    template<typename CloudT, typename PointT>
    CloudT getLidarPointCloud(int cloudIdx);
};


class ColoradarDataset {
public:
    ColoradarDataset(const std::filesystem::path& coloradarPath);

    Eigen::Affine3f getBaseToLidarTransform();
    Eigen::Affine3f getBaseToRadarTransform();
    std::vector<std::string> listRuns();
    ColoradarRun getRun(const std::string& runName);

    void createMaps(
        const double& mapResolution,
        const float& lidarTotalHorizontalFov,
        const float& lidarTotalVerticalFov,
        const float& lidarMaxRange,
        const std::vector<std::string>& targetRuns = std::vector<std::string>()
    );

protected:
    std::filesystem::path coloradarDirPath;
    std::filesystem::path calibDirPath;
    std::filesystem::path transformsDirPath;
    std::filesystem::path runsDirPath;

    Eigen::Affine3f loadTransform(const std::filesystem::path& filePath);
};

}

#endif
