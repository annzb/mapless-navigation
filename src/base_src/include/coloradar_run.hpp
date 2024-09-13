#ifndef COLORADAR_RUN_HPP
#define COLORADAR_RUN_HPP

#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <fstream>
#include <sstream>


namespace {

    template<typename PoseT>
    PoseT readPose(std::istringstream* iss);

    template<>
    octomath::Pose6D readPose<octomath::Pose6D>(std::istringstream* iss) {
        octomath::Vector3 translation;
        octomath::Quaternion rotation;
        *iss >> translation.x() >> translation.y() >> translation.z() >> rotation.x() >> rotation.y() >> rotation.z() >> rotation.u();
        return octomath::Pose6D(translation, rotation);
    }
    template<>
    Eigen::Affine3f readPose<Eigen::Affine3f>(std::istringstream* iss) {
        Eigen::Vector3f translation;
        Eigen::Quaternionf rotation;
        *iss >> translation.x() >> translation.y() >> translation.z() >> rotation.x() >> rotation.y() >> rotation.z() >> rotation.w();
        Eigen::Affine3f pose;
        pose.translate(translation);
        pose.rotate(rotation);
        return pose;
    }

    template<coloradar::Pcl4dPointType PointT>
    PointT makePoint(const float& x, const float& y, const float& z, const float& i) { return PointT(x, y, z, i); }

    template<coloradar::PointType PointT>
    PointT makePoint(const float& x, const float& y, const float& z, const float& i) { return PointT(x, y, z); }

}

template<typename PoseT>
std::vector<PoseT> coloradar::ColoradarRun::getPoses() {
    std::filesystem::path posesFilePath = posesDirPath / "groundtruth_poses.txt";
    coloradar::internal::checkPathExists(posesFilePath);

    std::vector<PoseT> poses;
    std::ifstream infile(posesFilePath);
    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        float x, y, z;
        PoseT pose = readPose<PoseT>(&iss);
        poses.push_back(pose);
    }
    return poses;
}

template<coloradar::CloudType CloudT>
CloudT coloradar::ColoradarRun::getLidarPointCloud(const std::filesystem::path& binPath) {
     // using PointT = std::conditional_t<PclCloudType<CloudT>, typename CloudT::PointType, octomap::point3d>;
     using PointT = std::conditional_t<PclCloudType<CloudT>, pcl::PointXYZI, octomap::point3d>;

    coloradar::internal::checkPathExists(binPath);
    std::ifstream infile(binPath, std::ios::binary);
    if (!infile) {
        throw std::runtime_error("Failed to open file: " + binPath.string());
    }
    infile.seekg(0, std::ios::end);
    size_t numPoints = infile.tellg() / (4 * sizeof(float));
    infile.seekg(0, std::ios::beg);

    CloudT cloud;
    cloud.reserve(numPoints);

    for (size_t j = 0; j < numPoints; ++j) {
        float x, y, z, i;
        infile.read(reinterpret_cast<char*>(&x), sizeof(float));
        infile.read(reinterpret_cast<char*>(&y), sizeof(float));
        infile.read(reinterpret_cast<char*>(&z), sizeof(float));
        infile.read(reinterpret_cast<char*>(&i), sizeof(float));
        cloud.push_back(makePoint<PointT>(x, y, z, i));
    }
    if (cloud.size() < 1) {
        throw std::runtime_error("Failed to read or empty point cloud: " + binPath.string());
    }
    return cloud;
}

template<coloradar::CloudType CloudT>
CloudT coloradar::ColoradarRun::getLidarPointCloud(const int& cloudIdx) {
    std::filesystem::path pclBinFilePath = pointcloudsDirPath / ("lidar_pointcloud_" + std::to_string(cloudIdx) + ".bin");
    return getLidarPointCloud<CloudT>(pclBinFilePath);
}

#endif
