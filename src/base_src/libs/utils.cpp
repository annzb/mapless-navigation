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

std::filesystem::path coloradar::internal::replaceInFilename(const std::filesystem::path& originalPath, const std::string& toReplace, const std::string& replacement) {
    std::filesystem::path newPath = originalPath.parent_path();
    std::string filename = originalPath.filename().string();
    std::size_t pos = filename.find(toReplace);
    if (pos != std::string::npos) {
        filename.replace(pos, toReplace.length(), replacement);
    }
    newPath /= filename;
    return newPath;
}


template<> octomath::Vector3 coloradar::internal::fromEigenTrans(const Eigen::Vector3f& r) { return octomath::Vector3(r.x(), r.y(), r.z()); }
template<> octomath::Quaternion coloradar::internal::fromEigenQuat(const Eigen::Quaternionf& r) { return octomath::Quaternion(r.w(), r.x(), r.y(), r.z()); }

Eigen::Vector3f coloradar::internal::sphericalToCartesian(const double& azimuth, const double& elevation, const double& range) {
    float x = range * cos(elevation) * cos(azimuth);
    float y = range * cos(elevation) * sin(azimuth);
    float z = range * sin(elevation);
    return Eigen::Vector3f(x, y, z);
}
