#ifndef UTILS_H
#define UTILS_H

#include "types.h"

#include <filesystem>


namespace coloradar::internal {

    void checkPathExists(const std::filesystem::path& path);
    void createDirectoryIfNotExists(const std::filesystem::path& dirPath);
    std::filesystem::path replaceInFilename(const std::filesystem::path& originalPath, const std::string& toReplace, const std::string& replacement);

    template<coloradar::Pcl4dPointType PointT> PointT makePoint(const float& x, const float& y, const float& z, const float& i);
    template<coloradar::PointType PointT> PointT makePoint(const float& x, const float& y, const float& z, const float& i);

    template<coloradar::PclPoseType PoseT>
    PoseT makePose(const typename coloradar::PoseTraits<PoseT>::TranslationType& translation, const typename coloradar::PoseTraits<PoseT>::RotationType& rotation);
    template<coloradar::OctoPoseType PoseT>
    PoseT makePose(const typename coloradar::PoseTraits<PoseT>::TranslationType& translation, const typename coloradar::PoseTraits<PoseT>::RotationType& rotation);

    template<coloradar::PclPoseType PoseT> Eigen::Vector3f toEigenTrans(const PoseT& pose);
    template<coloradar::PclPoseType PoseT> Eigen::Quaternionf toEigenQuat(const PoseT& pose);
    template<coloradar::OctoPoseType PoseT> Eigen::Vector3f toEigenTrans(const PoseT& pose);
    template<coloradar::OctoPoseType PoseT> Eigen::Quaternionf toEigenQuat(const PoseT& pose);
;
    template<typename TransT> TransT fromEigenTrans(const Eigen::Vector3f& r);
    template<typename RotationT> RotationT fromEigenQuat(const Eigen::Quaternionf& r);
    template<> octomath::Vector3 fromEigenTrans(const Eigen::Vector3f& r);
    template<> octomath::Quaternion fromEigenQuat(const Eigen::Quaternionf& r);

    template<coloradar::PclPoseType PoseT> Eigen::Affine3f toEigenPose(const PoseT& pose);
    template<coloradar::PclPoseType PoseT> PoseT fromEigenPose(const Eigen::Affine3f& pose);
    template<coloradar::OctoPoseType PoseT> Eigen::Affine3f toEigenPose(const PoseT& pose);
    template<coloradar::OctoPoseType PoseT> PoseT fromEigenPose(const Eigen::Affine3f& pose);

    template<coloradar::PointType PointT, coloradar::CloudType CloudT> CloudT readLidarPointCloud(const std::filesystem::path& binPath);

    template<typename PointT, typename CloudT> void filterFov(CloudT& cloud, const float& horizontalFov, const float& verticalFov, const float& range);

    Eigen::Vector3f sphericalToCartesian(const double& az, const double& el, const double& range);
}

#include "utils.hpp"

#endif
