#include "utils.h"
#include "coloradar_tools.h"


pcl::PointCloud<pcl::PointXYZI> coloradar::octreeToPcl(const octomap::OcTree& tree) {
    pcl::PointCloud<pcl::PointXYZI> cloud;
    for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
        octomap::point3d coords = it.getCoordinate();
        pcl::PointXYZI point;
        point.x = coords.x();
        point.y = coords.y();
        point.z = coords.z();
        point.intensity = it->getLogOdds();
        cloud.push_back(point);
    }
    return cloud;
}


template<typename PointT>
void coloradar::filterFov(pcl::PointCloud<PointT>& cloud, const float& horizontalFov, const float& verticalFov, const float& range) {
    return coloradar_utils::filterFov<pcl::PointCloud<PointT>, PointT>(cloud, horizontalFov, verticalFov, range);
}
template void coloradar::filterFov<pcl::PointXYZ>(pcl::PointCloud<pcl::PointXYZ>& cloud, const float& horizontalFov, const float& verticalFov, const float& range);
template void coloradar::filterFov<pcl::PointXYZI>(pcl::PointCloud<pcl::PointXYZI>& cloud, const float& horizontalFov, const float& verticalFov, const float& range);
