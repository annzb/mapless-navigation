#ifndef COLORADAR_RUN_HPP
#define COLORADAR_RUN_HPP

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <fstream>
#include <sstream>


template<coloradar::PoseType PoseT>
std::vector<PoseT> coloradar::ColoradarRun::getPoses() {
    std::filesystem::path posesFilePath = posesDirPath / "groundtruth_poses.txt";
    coloradar::internal::checkPathExists(posesFilePath);

    std::vector<PoseT> poses;
    std::ifstream infile(posesFilePath);
    std::string line;

    while (std::getline(infile, line)) {
        float x, y, z, rotX, rotY, rotZ, rotW;
        std::istringstream iss(line);
        iss >> x >> y >> z >> rotX >> rotY >> rotZ >> rotW;
        typename coloradar::PoseTraits<PoseT>::TranslationType translation(x, y, z);
        typename coloradar::PoseTraits<PoseT>::RotationType rotation(rotW, rotX, rotY, rotZ);
        PoseT pose = coloradar::internal::makePose<PoseT>(translation, rotation);
        poses.push_back(pose);
    }
    return poses;
}

template<coloradar::PoseType PoseT>
std::vector<PoseT> coloradar::ColoradarRun::interpolatePoses(const std::vector<PoseT>& poses, const std::vector<double>& poseTimestamps, const std::vector<double>& targetTimestamps) {
    size_t tsStartIdx = 0, targetTsStartIdx = 0;
    size_t tsEndIdx = poseTimestamps.size() - 1, targetTsEndIdx = targetTimestamps.size() - 1;
    if (poses.size() == 1) {
        std::vector<PoseT> interpolatedPoses(targetTsEndIdx - targetTsStartIdx + 1, poses[0]);
        return interpolatedPoses;
    }
    while (targetTsStartIdx < targetTsEndIdx && targetTimestamps[targetTsStartIdx] < poseTimestamps[tsStartIdx])
        targetTsStartIdx++;
    while (targetTsEndIdx > targetTsStartIdx && targetTimestamps[targetTsEndIdx] > poseTimestamps[tsEndIdx])
        targetTsEndIdx--;

    std::vector<PoseT> interpolatedPoses;
    size_t tsIdx = tsStartIdx;

    for (size_t targetTsIdx = targetTsStartIdx; targetTsIdx <= targetTsEndIdx; ++targetTsIdx) {
        while (tsIdx + 1 <= tsEndIdx && poseTimestamps[tsIdx + 1] < targetTimestamps[targetTsIdx])
            tsIdx++;
        double denominator = poseTimestamps[tsIdx + 1] - poseTimestamps[tsIdx];
        double ratio = denominator > 0.0f ? (targetTimestamps[targetTsIdx] - poseTimestamps[tsIdx]) / denominator : 0.0f;

        auto interpolatedPos = (1.0f - ratio) * poses[tsIdx].translation() + ratio * poses[tsIdx + 1].translation();
        auto interpolatedRot = poses[tsIdx].rotation().slerp(ratio, poses[tsIdx + 1].rotation());
        PoseT interpolatedPose = coloradar::internal::makePose<PoseT>(interpolatedPos, interpolatedRot.toRotationMatrix());
        interpolatedPoses.push_back(interpolatedPose);
    }
    return interpolatedPoses;
}

template<coloradar::PclCloudType CloudT>
CloudT coloradar::ColoradarRun::getLidarPointCloud(const std::filesystem::path& binPath) {
    return coloradar::internal::readLidarPointCloud<typename CloudT::PointType, CloudT>(binPath);
}

template<coloradar::OctomapCloudType CloudT>
CloudT coloradar::ColoradarRun::getLidarPointCloud(const std::filesystem::path& binPath) {
    return coloradar::internal::readLidarPointCloud<octomap::point3d, CloudT>(binPath);
}

template<coloradar::CloudType CloudT>
CloudT coloradar::ColoradarRun::getLidarPointCloud(const int& cloudIdx) {
    std::filesystem::path pclBinFilePath = pointcloudsDirPath / ("lidar_pointcloud_" + std::to_string(cloudIdx) + ".bin");
    return getLidarPointCloud<CloudT>(pclBinFilePath);
}

#endif
