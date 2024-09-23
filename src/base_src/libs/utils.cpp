#include "utils.h"

#include <stdexcept>


void coloradar::internal::checkPathExists(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("Directory or file not found: " + path.string());
    }
}
void coloradar::internal::createDirectoryIfNotExists(const std::filesystem::path& dirPath) {
    if (!std::filesystem::exists(dirPath)) {
        std::filesystem::create_directories(dirPath);
    }
}

template<> Eigen::Vector3f coloradar::internal::toEigenTrans(const octomath::Pose6D& pose) { return Eigen::Vector3f(pose.trans().x(), pose.trans().y(), pose.trans().z()); }
template<> Eigen::Quaternionf coloradar::internal::toEigenQuat(const octomath::Pose6D& pose) { return Eigen::Quaternionf(pose.rot().u(), pose.rot().x(), pose.rot().y(), pose.rot().z()); }

template<> octomath::Vector3 coloradar::internal::fromEigenTrans(const Eigen::Vector3f& r) { return octomath::Vector3(r.x(), r.y(), r.z()); }
template<> octomath::Quaternion coloradar::internal::fromEigenQuat(const Eigen::Quaternionf& r) { return octomath::Quaternion(r.w(), r.x(), r.y(), r.z()); }
