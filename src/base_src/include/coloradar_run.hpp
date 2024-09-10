#ifndef COLORADAR_RUN_HPP
#define COLORADAR_RUN_HPP

// #include "utils.cpp"

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

    template<typename PointT>
    PointT readLidarPoint(std::ifstream& infile) {
        float x, y, z;
        infile.read(reinterpret_cast<char*>(&x), sizeof(float));
        infile.read(reinterpret_cast<char*>(&y), sizeof(float));
        infile.read(reinterpret_cast<char*>(&z), sizeof(float));
        infile.ignore(sizeof(float));
        return PointT(x, y, z);
    }
    template<coloradar::Pcl4dPointType PointT>
    PointT readLidarPoint(std::ifstream& infile) {
        float x, y, z, i;
        infile.read(reinterpret_cast<char*>(&x), sizeof(float));
        infile.read(reinterpret_cast<char*>(&y), sizeof(float));
        infile.read(reinterpret_cast<char*>(&z), sizeof(float));
        infile.read(reinterpret_cast<char*>(&i), sizeof(float));
        return PointT(x, y, z, i);
    }

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
        PoseT pose = readPose<PoseT>(&iss);
        poses.push_back(pose);
    }
    return poses;
}

template<typename PointT, typename CloudT>
CloudT coloradar::ColoradarRun::getLidarPointCloud(const std::filesystem::path& binPath) {
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

    for (size_t i = 0; i < numPoints; ++i) {
        PointT point = readLidarPoint<PointT>(infile);
        cloud.push_back(readLidarPoint);
    }
    if (cloud.size() < 1) {
        throw std::runtime_error("Failed to read or empty point cloud: " + binPath.string());
    }
    return cloud;
}

template<typename PointT, typename CloudT>
CloudT coloradar::ColoradarRun::getLidarPointCloud(int cloudIdx) {
    std::filesystem::path pclBinFilePath = pointcloudsDirPath / ("lidar_pointcloud_" + std::to_string(cloudIdx) + ".bin");
    return getLidarPointCloud<PointT, CloudT>(pclBinFilePath);
}

}

#endif
