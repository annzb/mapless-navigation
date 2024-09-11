#include <gtest/gtest.h>
#include <cmath>
#include <random>
#include <pcl/io/pcd_io.h>

#include "coloradar_tools.h"


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


class CompileTest : public ::testing::Test {
public:
    CompileTest() : gen(randomSeed) {}

    pcl::PointCloud<pcl::PointXYZ> create3dPcl() {
        pcl::PointCloud<pcl::PointXYZ> cloud;
        cloud.points.push_back(pcl::PointXYZ(0.f, 0.f, 0.f));
        cloud.points.push_back(pcl::PointXYZ(1.f, 1.f, 1.f));
        cloud.points.push_back(pcl::PointXYZ(2.f, 1.f, 0.f));
        cloud.points.push_back(pcl::PointXYZ(1.f, 5.f, 0.f));
        cloud.points.push_back(pcl::PointXYZ(0.f, 1.f, 2.f));
        cloud.points.push_back(pcl::PointXYZ(-2.f, 1.f, 3.f));
        cloud.points.push_back(pcl::PointXYZ(10.f, 10.f, 10.f));
        return cloud;
    }

    pcl::PointCloud<pcl::PointXYZI> create4dPcl() {
        pcl::PointCloud<pcl::PointXYZI> cloud;
        cloud.points.push_back(pcl::PointXYZI(0.f, 0.f, 0.f, generateIntensity()));
        cloud.points.push_back(pcl::PointXYZI(1.f, 1.f, 1.f, generateIntensity()));
        cloud.points.push_back(pcl::PointXYZI(2.f, 1.f, 0.f, generateIntensity()));
        cloud.points.push_back(pcl::PointXYZI(1.f, 5.f, 0.f, generateIntensity()));
        cloud.points.push_back(pcl::PointXYZI(0.f, 1.f, 2.f, generateIntensity()));
        cloud.points.push_back(pcl::PointXYZI(10.f, 10.f, 10.f, generateIntensity()));
        cloud.points.push_back(pcl::PointXYZI(-2.f, 1.f, 3.f, generateIntensity()));
        return cloud;
    }

protected:
    const float treeResolution = 0.25;
    const int randomSeed = 42;
    std::mt19937 gen;
    std::uniform_real_distribution<> dis{-1.0, 1.0};
    float generateIntensity() { return dis(gen) * 10; }
};


TEST_F(CompileTest, BasicFunctions) {
    octomap::OcTree tree(treeResolution);
    pcl::PointCloud<pcl::PointXYZI> cloud;
    coloradar::octreeToPcl(tree, cloud);
    coloradar::filterFov(cloud, 360, 180, 10);

//    Compile Error
//    pcl::PointCloud<pcl::PointXYZ> cloud3d;
//    coloradar::octreeToPcl(tree, cloud3d);
//    coloradar::filterFov(cloud3d, 360, 180, 10);
}

TEST_F(CompileTest, OctoPointcloud) {
    auto cloud3d = create3dPcl();
    coloradar::OctoPointcloud octoCloud(cloud3d);
    octoCloud.filterFov(360, 180, 10);
    auto otherCloud = octoCloud.toPcl<pcl::PointCloud<pcl::PointXYZI>>();
    auto cloud4d = create4dPcl();
    octoCloud = coloradar::OctoPointcloud(cloud4d);
}


//class PclFilterTest : public ::testing::Test {
//protected:
//    double nearEps = 0.01;
//};
//
//TEST_F(PclFilterTest, RangeTest) {
//    pcl::PointCloud<pcl::PointXYZ> cloud = createPcl();
//    std::cout << "Cloud before: " << cloud.size() << " points,";
//    for (size_t i = 0; i < cloud.size(); ++i) {
//        std::cout << " " << cloud.points[i];
//    }
//    std::cout << std::endl;
//
//    coloradar::filterFov(cloud, 360, 360, 10);
//    std::cout << "Cloud after range: " << cloud.size() << " points,";
//    for (size_t i = 0; i < cloud.size(); ++i) {
//        std::cout << " " << cloud.points[i];
//    }
//    std::cout << std::endl;
//
//    coloradar::filterFov(cloud, 90, 360, 0);
//    std::cout << "Cloud after azimuth:";
//    for (size_t i = 0; i < cloud.size(); ++i) {
//        std::cout << " " << cloud.points[i];
//    }
//    std::cout << std::endl;
//
//    coloradar::filterFov(cloud, 360, 90, 0);
//    std::cout << "Cloud after elevation:";
//    for (size_t i = 0; i < cloud.size(); ++i) {
//        std::cout << " " << cloud.points[i];
//    }
//    std::cout << std::endl;
//
//    cloud = createPcl();
//    coloradar::filterFov(cloud, 90, 90, 10);
//    std::cout << "Cloud filtered:";
//    for (size_t i = 0; i < cloud.size(); ++i) {
//        std::cout << " " << cloud.points[i];
//    }
//    std::cout << std::endl;
//}


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
