#include "utils.h"
#include "coloradar_tools.h"


coloradar::OctoPointcloud::OctoPointcloud(const pcl::PointCloud<pcl::PointXYZ>& cloud) {
    this->clear();
    this->reserve(cloud.size());
    for (const auto& point : cloud.points) {
        this->push_back(octomap::point3d(point.x, point.y, point.z));
    }
}

pcl::PointCloud<pcl::PointXYZ> coloradar::OctoPointcloud::toPcl() {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.reserve(this->size());
    for (size_t i = 0; i < this->size(); ++i) {
        const octomap::point3d& point = this->getPoint(i);
        cloud.push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
    }
    return cloud;
}

void coloradar::OctoPointcloud::transform(const Eigen::Affine3f& transformMatrix) {
    Eigen::Quaternionf rotation(transformMatrix.rotation());
    octomath::Pose6D transformPose(
        octomath::Vector3(transformMatrix.translation().x(), transformMatrix.translation().y(), transformMatrix.translation().z()),
        octomath::Quaternion(rotation.w(), rotation.x(), rotation.y(), rotation.z())
    );
    this->transform(transformPose);
}

void coloradar::OctoPointcloud::filterFov(const float& horizontalFov, const float& verticalFov, const float& range) {
    coloradar_utils::filterFov<coloradar::OctoPointcloud, octomap::point3d>(*this, horizontalFov, verticalFov, range);
}
