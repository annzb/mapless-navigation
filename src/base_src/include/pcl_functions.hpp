#ifndef PCL_FUNCTIONS_HPP
#define PCL_FUNCTIONS_HPP


template <coloradar::Pcl4dPointType PointT, template <coloradar::PclCloudType> class CloudT>
void coloradar::octreeToPcl(const octomap::OcTree& tree, CloudT<PointT>& cloud) {
    cloud.clear();
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

template <coloradar::PclPointType PointT, template <coloradar::PclCloudType> class CloudT>
void coloradar::filterFov(CloudT<PointT>& cloud, const float& horizontalFov, const float& verticalFov, const float& range) {
    coloradar::internal::filterFov<PointT, CloudT<PointT>>(cloud, horizontalFov, verticalFov, range);
}

#endif
