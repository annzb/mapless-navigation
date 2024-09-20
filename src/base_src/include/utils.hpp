#ifndef UTILS_HPP
#define UTILS_HPP

#include <stdexcept>
#include <functional>


namespace {

    template<typename PointT>
    float getX(const PointT& point) { return point.x; }
    template<coloradar::OctomapPointType PointT>
    float getX(const PointT& point) { return point.x(); }

    template<typename PointT>
    float getY(const PointT& point) { return point.y; }
    template<coloradar::OctomapPointType PointT>
    float getY(const PointT& point) { return point.y(); }

    template<typename PointT>
    float getZ(const PointT& point) { return point.z; }
    template<coloradar::OctomapPointType PointT>
    float getZ(const PointT& point) { return point.z(); }

    template<typename PointT>
    bool checkAzimuthFrontOnly(const PointT& point, const float& horizontalHalfFovRad) {
        float horizontalFovTan = std::tan(horizontalHalfFovRad);
        return (getY(point) >= getX(point) / horizontalFovTan && getY(point) >= -getX(point) / horizontalFovTan);
    }
    template<typename PointT>
    bool checkAzimuthFrontBack(const PointT& point, const float& horizontalHalfFovRad) {
        float horizontalFovTan = std::tan(horizontalHalfFovRad);
        return (getY(point) >= getX(point) / horizontalFovTan || getY(point) >= -getX(point) / horizontalFovTan);
    }

    template<typename PointT>
    using FovCheck = std::function<bool(const PointT&, const float&)>;

}

template<coloradar::Pcl4dPointType PointT>
PointT coloradar::internal::makePoint(const float& x, const float& y, const float& z, const float& i) { return PointT(x, y, z, i); }

template<coloradar::PointType PointT>
PointT coloradar::internal::makePoint(const float& x, const float& y, const float& z, const float& i) { return PointT(x, y, z); }

template<coloradar::PclPoseType PoseT>
PoseT coloradar::internal::makePose(const typename coloradar::PoseTraits<PoseT>::TranslationType& translation, const typename coloradar::PoseTraits<PoseT>::RotationType& rotation) {
    PoseT pose = PoseT::Identity();
    pose.translate(translation);
    pose.rotate(rotation);
    return pose;
}
template<coloradar::OctoPoseType PoseT>
PoseT coloradar::internal::makePose(const typename coloradar::PoseTraits<PoseT>::TranslationType& translation, const typename coloradar::PoseTraits<PoseT>::RotationType& rotation) {
    return PoseT(translation, rotation);
}


template<coloradar::PointType PointT, coloradar::CloudType CloudT>
CloudT coloradar::internal::readLidarPointCloud(const std::filesystem::path& binPath) {
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
        cloud.push_back(coloradar::internal::makePoint<PointT>(x, y, z, i));
    }
    if (cloud.size() < 1) {
        throw std::runtime_error("Failed to read or empty point cloud: " + binPath.string());
    }
    return cloud;
}


template<typename PointT, typename CloudT>
void coloradar::internal::filterFov(CloudT& cloud, const float& horizontalFov, const float& verticalFov, const float& range) {
    if (horizontalFov <= 0 || horizontalFov > 360) {
        throw std::runtime_error("Invalid horizontal FOV value: expected 0 < FOV <= 360, got " + std::to_string(horizontalFov));
    }
    if (verticalFov <= 0 || verticalFov > 180) {
        throw std::runtime_error("Invalid vertical FOV value: expected 0 < FOV <= 180, got " + std::to_string(verticalFov));
    }
    if (range <= 0) {
        throw std::runtime_error("Invalid max range value: expected R > 0, got " + std::to_string(range));
    }
    float horizontalHalfFovRad = horizontalFov / 2 * M_PI / 180.0f;
    float verticalHalfFovRad = verticalFov / 2 * M_PI / 180.0f;
    FovCheck<PointT> checkAzimuth = horizontalFov <= 180 ? checkAzimuthFrontOnly<PointT> : checkAzimuthFrontBack<PointT>;

    CloudT unfilteredCloud(cloud);
    cloud.clear();

    for (size_t i = 0; i < unfilteredCloud.size(); ++i) {
        const PointT& point = unfilteredCloud[i];
        float distance = std::sqrt(std::pow(getX(point), 2) + std::pow(getY(point), 2) + std::pow(getZ(point), 2));
        if (distance > range) {
            continue;
        }
        float elevationSin = std::sin(verticalHalfFovRad);
        if (checkAzimuth(point, horizontalHalfFovRad) &&
           (getZ(point) <= distance * elevationSin) &&
           (getZ(point) >= -distance * elevationSin)
        ) {
            cloud.push_back(point);
        }
    }
}

#endif
