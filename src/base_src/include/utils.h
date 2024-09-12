#ifndef UTILS_H
#define UTILS_H

#include "types.h"

#include <filesystem>


namespace coloradar::internal {

    void checkPathExists(const std::filesystem::path& path);
    void createDirectoryIfNotExists(const std::filesystem::path& dirPath);

    template<typename PointT, typename CloudT>
    void filterFov(CloudT& cloud, const float& horizontalFov, const float& verticalFov, const float& range);

}

#include "utils.hpp"

#endif
