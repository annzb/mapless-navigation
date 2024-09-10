#ifndef OCTO_POINTCLOUD_HPP
#define OCTO_POINTCLOUD_HPP

// #include "utils.cpp"


template<coloradar::PclCloudType<coloradar::PclPointType> CloudT>
coloradar::OctoPointcloud::OctoPointcloud(const CloudT& cloud) {
    this->clear();
    this->reserve(cloud.size());
    for (const auto& point : cloud.points) {
        this->push_back(octomap::point3d(point.x, point.y, point.z));
    }
}

template<coloradar::PclCloudType<coloradar::PclPointType> CloudT>
CloudT coloradar::OctoPointcloud::toPcl() {
    using PointT = typename CloudT::PointType;
    CloudT cloud;
    cloud.reserve(this->size());
    for (size_t i = 0; i < this->size(); ++i) {
        const octomap::point3d& point = this->getPoint(i);
        cloud.push_back(PointT(point.x(), point.y(), point.z()));
    }
    return cloud;
}

#endif
