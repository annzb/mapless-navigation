#ifndef PCL_FUNCTIONS_HPP
#define PCL_FUNCTIONS_HPP

// #include "utils.cpp"


template<coloradar::PclCloudType<coloradar::Pcl4dPointType> CloudT>
void coloradar::octreeToPcl(const octomap::OcTree& tree, CloudT& cloud) {
    using PointT = typename CloudT::PointType;
    for (auto it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it) {
        PointT point;
        octomap::point3d coords = it.getCoordinate();
        point.x = coords.x();
        point.y = coords.y();
        point.z = coords.z();
        point.intensity = it->getLogOdds();
        cloud.push_back(point);
    }
}

template<coloradar::PclCloudType<coloradar::PclPointType> CloudT>
void coloradar::filterFov(CloudT& cloud, const float& horizontalFov, const float& verticalFov, const float& range) {
    using PointT = typename CloudT::PointType;
    coloradar::internal::filterFov<PointT, CloudT>(cloud, horizontalFov, verticalFov, range);
}

#endif
