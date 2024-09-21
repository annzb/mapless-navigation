#ifndef UTILS_H
#define UTILS_H

#include "types.h"

#include <filesystem>


namespace coloradar::internal {

    void checkPathExists(const std::filesystem::path& path);
    void createDirectoryIfNotExists(const std::filesystem::path& dirPath);

    template<typename RotationT> Eigen::Quaternionf toEigenQuat(const RotationT& r);
    template<typename RotationT> RotationT fromEigenQuat(const Eigen::Quaternionf& r);

    template<coloradar::Pcl4dPointType PointT> PointT makePoint(const float& x, const float& y, const float& z, const float& i);
    template<coloradar::PointType PointT> PointT makePoint(const float& x, const float& y, const float& z, const float& i);

    template<coloradar::PclPoseType PoseT>
    PoseT makePose(const typename coloradar::PoseTraits<PoseT>::TranslationType& translation, const typename coloradar::PoseTraits<PoseT>::RotationType& rotation);
    template<coloradar::OctoPoseType PoseT>
    PoseT makePose(const typename coloradar::PoseTraits<PoseT>::TranslationType& translation, const typename coloradar::PoseTraits<PoseT>::RotationType& rotation);

    template<coloradar::PointType PointT, coloradar::CloudType CloudT> CloudT readLidarPointCloud(const std::filesystem::path& binPath);

    template<typename PointT, typename CloudT> void filterFov(CloudT& cloud, const float& horizontalFov, const float& verticalFov, const float& range);
}

#include "utils.hpp"

#endif
