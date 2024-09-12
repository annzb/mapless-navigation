#ifndef COLORADAR_TOOLS_H
#define COLORADAR_TOOLS_H

#include "utils.h"

#include <Eigen/Dense>
#include <string>
#include <vector>


namespace coloradar {

template <Pcl4dPointType PointT, template <PclCloudType> class CloudT> void octreeToPcl(const octomap::OcTree& tree, CloudT<PointT>& cloud);
template <PclPointType PointT, template <PclCloudType> class CloudT> void filterFov(CloudT<PointT>& cloud, const float& horizontalFov, const float& verticalFov, const float& range);


class OctoPointcloud : public octomap::Pointcloud {
public:
    OctoPointcloud() = default;
    OctoPointcloud(const OctoPointcloud& other) : octomap::Pointcloud(other) {}
    template <PclPointType PointT, template <PclCloudType> class CloudT> OctoPointcloud(const CloudT<PointT>& cloud);

    template <PclCloudType CloudT> CloudT toPcl();

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

    template<typename PoseT> std::vector<PoseT> getPoses();
    template<CloudType CloudT> CloudT getLidarPointCloud(const std::filesystem::path& binPath);
    template<CloudType CloudT> CloudT getLidarPointCloud(const int& cloudIdx);

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

#include "pcl_functions.hpp"
#include "octo_pointcloud.hpp"
#include "coloradar_run.hpp"

#endif
