#ifndef UTILS_H
#define UTILS_H

#include <filesystem>


void checkPathExists(const std::filesystem::path& path);
void createDirectoryIfNotExists(const std::filesystem::path& dirPath);

template<typename CloudT, typename PointT>
void filterFov(CloudT& cloud, const float& horizontalFov, const float& verticalFov, const float& range);

#endif
