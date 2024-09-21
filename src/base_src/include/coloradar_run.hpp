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
    std::vector<PoseT> interpolatedPoses;
    size_t tsStartIdx = 0, tsEndIdx = poseTimestamps.size() - 1;
    size_t tsIdx = tsStartIdx;

    for (size_t targetTsIdx = 0; targetTsIdx < targetTimestamps.size(); ++targetTsIdx) {
        if (targetTimestamps[targetTsIdx] < poseTimestamps[tsStartIdx]) {
            interpolatedPoses.push_back(poses[tsStartIdx]);
            continue;
        }
        if (targetTimestamps[targetTsIdx] > poseTimestamps[tsEndIdx]) {
            interpolatedPoses.push_back(poses[tsEndIdx]);
            continue;
        }
        while (tsIdx + 1 <= tsEndIdx && poseTimestamps[tsIdx + 1] < targetTimestamps[targetTsIdx])
            tsIdx++;
        double denominator = poseTimestamps[tsIdx + 1] - poseTimestamps[tsIdx];
        double ratio = denominator > 0.0f ? (targetTimestamps[targetTsIdx] - poseTimestamps[tsIdx]) / denominator : 0.0f;
        auto interpolatedPos = (1.0f - ratio) * poses[tsIdx].translation() + ratio * poses[tsIdx + 1].translation();
        Eigen::Quaternionf q1 = coloradar::internal::toEigenQuat(poses[tsIdx].rotation());
        Eigen::Quaternionf q2 = coloradar::internal::toEigenQuat(poses[tsIdx + 1].rotation());
        if (q1.dot(q2) < 0.0f) q2.coeffs() = -q2.coeffs();
        q1.normalize(); q2.normalize();
        Eigen::Quaternionf interpolatedRotEig = q1.slerp(ratio, q2);
        // if (targetTsIdx > targetTimestamps.size() / 2)
        //     std::cout << "q1: " << q1.coeffs().transpose() << " q2: " << q2.coeffs().transpose() << " interpolated: " << interpolatedRotEig.coeffs().transpose() << std::endl;
        interpolatedRotEig.normalize();
        auto interpolatedRot = coloradar::internal::fromEigenQuat<typename coloradar::PoseTraits<PoseT>::RotationType>(interpolatedRotEig);
        PoseT interpolatedPose = coloradar::internal::makePose<PoseT>(interpolatedPos, interpolatedRot);
        interpolatedPoses.push_back(interpolatedPose);
    }
    return interpolatedPoses;
}

//template<coloradar::PoseType PoseT>
//std::vector<PoseT> coloradar::ColoradarRun::interpolatePoses(const std::vector<PoseT>& poses, const std::vector<double>& poseTimestamps, const std::vector<double>& targetTimestamps) {
//    std::vector<PoseT> interpolatedPoses;
//    size_t tsEndIdx = poseTimestamps.size() - 1;
//
//    auto interpolateOrExtrapolate = [&](size_t idx1, size_t idx2, double targetTime) {
//        double denominator = poseTimestamps[idx2] - poseTimestamps[idx1];
//        double ratio = denominator > 0.0f ? (targetTime - poseTimestamps[idx1]) / denominator : 0.0f;
//        auto interpPos = (1.0f - ratio) * poses[idx1].translation() + ratio * poses[idx2].translation();
//        Eigen::Quaternionf interpRotEig = coloradar::internal::toEigenQuat(poses[idx1].rotation()).slerp(ratio, coloradar::internal::toEigenQuat(poses[idx2].rotation()));
//        auto interpRot = coloradar::internal::fromEigenQuat<typename coloradar::PoseTraits<PoseT>::RotationType>(interpRotEig);
//        return coloradar::internal::makePose<PoseT>(interpPos, interpRot);
//    };
//
//    size_t tsIdx = 0;
//    for (size_t targetTsIdx = 0; targetTsIdx < targetTimestamps.size(); ++targetTsIdx) {
//        double targetTime = targetTimestamps[targetTsIdx];
//        if (targetTime < poseTimestamps[0]) {
//            interpolatedPoses.push_back(interpolateOrExtrapolate(0, 1, targetTime));                   // Extrapolate before the first pose
//        } else if (targetTime > poseTimestamps[tsEndIdx]) {
//            interpolatedPoses.push_back(interpolateOrExtrapolate(tsEndIdx - 1, tsEndIdx, targetTime)); // Extrapolate after the last pose
//        } else {
//            while (tsIdx + 1 <= tsEndIdx && poseTimestamps[tsIdx + 1] < targetTime)
//                tsIdx++;
//            interpolatedPoses.push_back(interpolateOrExtrapolate(tsIdx, tsIdx + 1, targetTime));       // Interpolate between poses
//        }
//    }
//    return interpolatedPoses;
//}


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
    std::filesystem::path pclBinFilePath = lidarCloudsDirPath / ("lidar_pointcloud_" + std::to_string(cloudIdx) + ".bin");
    return getLidarPointCloud<CloudT>(pclBinFilePath);
}

#endif
