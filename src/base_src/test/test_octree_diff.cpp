#include <gtest/gtest.h>
#include <octomap/octomap.h>
#include <string>
#include <cmath>

#include "octree_diff.h"


octomap::point3d calculateVoxelCenter(double resolution, double x, double y, double z) {
    double half_res = resolution / 2.0;
    return octomap::point3d(
        std::round(x / resolution) * resolution + half_res,
        std::round(y / resolution) * resolution + half_res,
        std::round(z / resolution) * resolution + half_res
    );
}

octomap::OcTree createTestTree1(double resolution, double occupancy_threshold = 0.9) {
    octomap::OcTree tree(resolution);
    auto voxel1 = calculateVoxelCenter(resolution, 0.0, 0.0, 0.0);
    auto voxel2 = calculateVoxelCenter(resolution, 1.0, 1.0, 1.0);
    tree.updateNode(voxel1, false);  // Free leaf, prob around 0.1
    tree.search(voxel1)->setLogOdds(-2.2f);
    tree.updateNode(voxel2, true);  // Occupied leaf, prob around 0.95
    tree.search(voxel2)->setLogOdds(2.95f);
    return tree;
}

octomap::OcTree createTestTree2(double resolution, double occupancy_threshold = 0.9) {
    octomap::OcTree tree(resolution);
    auto voxel1 = calculateVoxelCenter(resolution, 0.0, 0.0, 0.0);
    auto voxel2 = calculateVoxelCenter(resolution, 1.0, 1.0, 1.0);
    tree.updateNode(voxel1, false);    // Free leaf, prob around 0.5
    tree.search(voxel1)->setLogOdds(0.f);
    tree.updateNode(voxel2, true); // Occupied leaf, prob around 0.95
    tree.search(voxel2)->setLogOdds(2.951f);
    return tree;
}


class OctreeDiffTest : public ::testing::Test {
protected:
    double nearEps = 0.01;
    double nodeDiffEps = 0.01;
    double treeResolution = 0.25;
};

TEST_F(OctreeDiffTest, DiffTrees) {
    octomap::OcTree tree1 = createTestTree1(treeResolution);
    octomap::OcTree tree2 = createTestTree2(treeResolution);
    auto diffNodes = calcOctreeDiff(tree1, tree2, nodeDiffEps);
//    std::cout << "Tree 1" << std::endl;
//    for (auto it = tree1.begin_leafs(), end = tree1.end_leafs(); it != end; ++it) {
//        std::cout << it.getCoordinate() << it->getLogOdds() << std::endl;
//    }
//    std::cout << "Tree 2" << std::endl;
//    for (auto it = tree2.begin_leafs(), end = tree2.end_leafs(); it != end; ++it) {
//        std::cout << it.getCoordinate() << it->getLogOdds() << std::endl;
//    }
//    std::cout << "Diff Nodes" << std::endl;
//    for (const auto& node : diffNodes) {
//        std::cout << node.coords << node.logOdds << node.logOddsDiff << std::endl;
//    }

    EXPECT_EQ(diffNodes.size(), 1);
    DiffNode diffNode = diffNodes.front();
    auto nodeInTree1 = tree1.search(diffNode.coords);
    if (!nodeInTree1) {
        FAIL() << "Unexpected node coordinates: " << diffNode.coords;
    }
    EXPECT_NEAR(diffNode.logOdds, 0.0, nearEps);
    EXPECT_NEAR(diffNode.logOddsDiff, 2.2, nearEps);
}

TEST_F(OctreeDiffTest, IdenticalTrees) {
    octomap::OcTree tree1 = createTestTree1(treeResolution);
    auto diffNodes = calcOctreeDiff(tree1, tree1, nodeDiffEps);
    EXPECT_EQ(diffNodes.size(), 0);
}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
