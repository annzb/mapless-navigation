#include <gtest/gtest.h>
#include <cmath>

#include "coloradar_tools.h"


pcl::PointCloud<pcl::PointXYZ> createPcl1() {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.points.push_back(pcl::PointXYZ(0, 0, 0));
    cloud.points.push_back(pcl::PointXYZ(1, 1, 1));
    cloud.points.push_back(pcl::PointXYZ(10, 10, 10));
    cloud.points.push_back(pcl::PointXYZ(2, 1, 0));
    cloud.points.push_back(pcl::PointXYZ(-2, 1, 3));
    cloud.points.push_back(pcl::PointXYZ(1, 5, 0));
    cloud.points.push_back(pcl::PointXYZ(0, 1, 2));
    return cloud;
}


class PclFilterTest : public ::testing::Test {
protected:
    double nearEps = 0.01;
};

TEST_F(PclFilterTest, RangeTest) {
    auto cloud = createPcl1();
    std::cout << "Cloud before: " << cloud.size() << " points,";
    for (size_t i = 0; i < cloud.size(); ++i) {
        std::cout << " " << cloud.points[i];
    }
    std::cout << std::endl;

    auto cloudInRange = filterFov(cloud, 360, 360, 10);
    std::cout << "Cloud after range: " << cloudInRange.size() << " points,";
    for (size_t i = 0; i < cloudInRange.size(); ++i) {
        std::cout << " " << cloudInRange.points[i];
    }
    std::cout << std::endl;

    auto cloudInAzimuth = filterFov(cloudInRange, 90, 360, 0);
    std::cout << "Cloud after azimuth:";
    for (size_t i = 0; i < cloudInAzimuth.size(); ++i) {
        std::cout << " " << cloudInAzimuth.points[i];
    }
    std::cout << std::endl;

    auto cloudInElevation = filterFov(cloudInAzimuth, 360, 90, 0);
    std::cout << "Cloud after elevation:";
    for (size_t i = 0; i < cloudInElevation.size(); ++i) {
        std::cout << " " << cloudInElevation.points[i];
    }
    std::cout << std::endl;

    auto cloudFiltered = filterFov(cloud, 90, 90, 10);
    std::cout << "Cloud filtered:";
    for (size_t i = 0; i < cloudFiltered.size(); ++i) {
        std::cout << " " << cloudFiltered.points[i];
    }
    std::cout << std::endl;
}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
