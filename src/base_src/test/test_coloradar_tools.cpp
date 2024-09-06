#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <pcl/io/pcd_io.h>

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

bool generateRandomEmptySpace(float probability) {
    static std::mt19937 gen(42);  // seed
    static std::uniform_real_distribution<> dis(0.0, 1.0);
    return dis(gen) < probability;
}

octomap::Pointcloud generateSpherePointCloud(float radius, float step, float emptySpaceProbability) {
    octomap::Pointcloud cloud;

    for (float x = -radius; x <= radius; x += step) {
        for (float y = -radius; y <= radius; y += step) {
            for (float z = -radius; z <= radius; z += step) {
                if (x*x + y*y + z*z <= radius * radius) {
                    if (!generateRandomEmptySpace(emptySpaceProbability)) {
                        cloud.push_back(octomap::point3d(x, y, z));
                    }
                }
            }
        }
    }
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

    coloradar::filterFov(cloud, 360, 360, 10);
    std::cout << "Cloud after range: " << cloud.size() << " points,";
    for (size_t i = 0; i < cloud.size(); ++i) {
        std::cout << " " << cloud.points[i];
    }
    std::cout << std::endl;

    coloradar::filterFov(cloud, 90, 360, 0);
    std::cout << "Cloud after azimuth:";
    for (size_t i = 0; i < cloud.size(); ++i) {
        std::cout << " " << cloud.points[i];
    }
    std::cout << std::endl;

    coloradar::filterFov(cloud, 360, 90, 0);
    std::cout << "Cloud after elevation:";
    for (size_t i = 0; i < cloud.size(); ++i) {
        std::cout << " " << cloud.points[i];
    }
    std::cout << std::endl;

    cloud = createPcl1();
    coloradar::filterFov(cloud, 90, 90, 10);
    std::cout << "Cloud filtered:";
    for (size_t i = 0; i < cloud.size(); ++i) {
        std::cout << " " << cloud.points[i];
    }
    std::cout << std::endl;
}


//void savePointCloudToPCD(const std::string& filename, const octomap::Pointcloud& cloud) {
//    pcl::PointCloud<pcl::PointXYZ> pclCloud;
//    for (size_t i = 0; i < cloud.size(); ++i) {
//        const octomap::point3d& point = cloud.getPoint(i);
//        pclCloud.push_back(pcl::PointXYZ(point.x(), point.y(), point.z()));
//    }
//    std::string homeDir = getenv("HOME");
//    pcl::io::savePCDFile(homeDir + "/" + filename + ".pcd", pclCloud);
//}
//
//TEST(FilterFovTest, FilterPointCloud) {
//    float horizontalFov = 90;
//    float verticalFov = 30;
//    float maxRange = 5;
//
//    float pclRadius = 10;
//    float pclStep = 0.5;
//    float emptySpaceProbability = 0.5;
//    OctoPointcloud cloud = generateSpherePointCloud(pclRadius, pclStep, emptySpaceProbability);
//    savePointCloudToPCD("original_cloud", cloud);
//
//    float horizontalFovTan = processFov(horizontalFov);
//    float verticalFovTan = processFov(verticalFov);
//    float range = processRange(maxRange);
//    filterFov(horizontalFovTan, verticalFovTan, range);
//    savePointCloudToPCD("filtered_cloud", filteredCloud);
//}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
