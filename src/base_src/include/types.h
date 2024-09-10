#ifndef TYPES_H
#define TYPES_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <octomap/octomap.h>


namespace coloradar {

template<typename T>
concept Pcl3dPointType = std::is_base_of_v<pcl::PointXYZ, T>;

template<typename T>
concept Pcl4dPointType = std::is_base_of_v<pcl::PointXYZI, T>;

template<typename T>
concept PclPointType = Pcl3dPointType<T> || Pcl4dPointType<T>;

template<typename T, typename PointT>
concept PclCloudType = requires {
    PclPointType<PointT>;
    std::is_base_of_v<pcl::PointCloud<PointT>, T>;
};

template<typename T>
concept OctomapPointType = std::is_base_of_v<octomap::point3d, T>;

template<typename T>
concept OctomapCloudType = std::is_base_of_v<octomap::Pointcloud, T>;

}

#endif