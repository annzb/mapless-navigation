#ifndef UTILS_HPP
#define UTILS_HPP

#include "types.h"

#include <filesystem>
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


namespace coloradar::internal {

    void checkPathExists(const std::filesystem::path& path) {
        if (!std::filesystem::exists(path)) {
            throw std::runtime_error("Directory or file not found: " + path.string());
        }
    }
    void createDirectoryIfNotExists(const std::filesystem::path& dirPath) {
        if (!std::filesystem::exists(dirPath)) {
            std::filesystem::create_directories(dirPath);
        }
    }

    template<typename PointT, typename CloudT>
    void filterFov(CloudT& cloud, const float& horizontalFov, const float& verticalFov, const float& range) {
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

}

#endif
