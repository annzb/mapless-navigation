#include <gtest/gtest.h>
#include <octomap/octomap.h>
#include <opencv2/opencv.hpp>
#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>
#include <string>
#include "octree_diff.h"

// Helper function to create a simple octree with more leaves
octomap::OcTree createSimpleTree(double resolution)
{
    octomap::OcTree tree(resolution);
    // Adding at least 3 leaves with different occupancy values
    tree.updateNode(octomap::point3d(0.0, 0.0, 0.0), true)->setLogOdds(0.9);  // Occupied leaf
    tree.updateNode(octomap::point3d(1.0, 1.0, 1.0), false)->setLogOdds(-0.9);  // Free leaf
    tree.updateNode(octomap::point3d(-1.0, -1.0, -1.0), true)->setLogOdds(0.5);  // Occupied leaf
    return tree;
}

// Helper function to save octree as an annotated image
void saveOctreeAsImage(const octomap::OcTree& tree, const std::string& filename, int image_size = 400)
{
    // Create a black image
    cv::Mat image = cv::Mat::zeros(image_size, image_size, CV_8UC3);

    for (octomap::OcTree::leaf_iterator it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it)
    {
        // Get coordinates and occupancy
        double x = it.getX();
        double y = it.getY();
        double log_odds = it->getLogOdds();
        bool occupied = tree.isNodeOccupied(*it);

        // Map the coordinates to image space
        int img_x = static_cast<int>((x + 2) * image_size / 4);  // Map x from [-2, 2] to [0, image_size]
        int img_y = static_cast<int>((y + 2) * image_size / 4);  // Map y from [-2, 2] to [0, image_size]

        // Set pixel color based on occupancy
        if (occupied)
        {
            cv::circle(image, cv::Point(img_x, img_y), 5, cv::Scalar(0, 255, 0), -1);  // Green for occupied
        }
        else
        {
            cv::circle(image, cv::Point(img_x, img_y), 5, cv::Scalar(0, 0, 255), -1);  // Red for free
        }

        // Annotate the image with occupancy log odds value
        std::string text = std::to_string(log_odds).substr(0, 4);  // Display only first 4 characters
        cv::putText(image, text, cv::Point(img_x + 10, img_y), cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
    }

    // Save the image
    cv::imwrite(filename, image);
}

// Helper function to create the output directory if it doesn't exist
void createDirectoryIfNotExists(const std::string& dir)
{
    struct stat info;
    if (stat(dir.c_str(), &info) != 0) {
        // Directory does not exist, create it
        if (mkdir(dir.c_str(), 0777) == -1) {
            std::cerr << "Error creating directory: " << dir << std::endl;
        } else {
            std::cout << "Directory created: " << dir << std::endl;
        }
    } else if (info.st_mode & S_IFDIR) {
        // Directory exists
        std::cout << "Directory already exists: " << dir << std::endl;
    } else {
        std::cerr << "Error: Path exists but is not a directory: " << dir << std::endl;
    }
}

class OctreeDiffTest : public ::testing::Test {
protected:
    std::string test_output_dir;

    void SetUp() override {
        const char* home = getenv("HOME");
        test_output_dir = std::string(home) + "/test_output";
        createDirectoryIfNotExists(test_output_dir);
    }
};

// Test for calcOctreeDiff with two identical trees
TEST_F(OctreeDiffTest, IdenticalTrees)
{
    // Create two identical trees
    octomap::OcTree tree1 = createSimpleTree(0.1);
    octomap::OcTree tree2 = createSimpleTree(0.1);

    // Save the trees as images
    saveOctreeAsImage(tree1, test_output_dir + "/identical_tree1.png");
    saveOctreeAsImage(tree2, test_output_dir + "/identical_tree2.png");

    // Calculate the diff
    auto [updateTree, diffTree] = calcOctreeDiff(tree1, tree2, 0.001);

    // Save the diff trees as images
    saveOctreeAsImage(updateTree, test_output_dir + "/identical_update_tree.png");
    saveOctreeAsImage(diffTree, test_output_dir + "/identical_diff_tree.png");

    // Check that both trees are empty (no differences)
    EXPECT_EQ(updateTree.size(), 0);
    EXPECT_EQ(diffTree.size(), 0);
}

// Test for calcOctreeDiff with trees that have differences
TEST_F(OctreeDiffTest, DifferentTrees)
{
    // Create two different trees
    octomap::OcTree tree1 = createSimpleTree(0.1);
    octomap::OcTree tree2 = createSimpleTree(0.1);
    tree2.updateNode(octomap::point3d(-1.0, -1.0, -1.0), true)->setLogOdds(0.9);  // Modify one leaf in tree2

    // Save the trees as images
    saveOctreeAsImage(tree1, test_output_dir + "/different_tree1.png");
    saveOctreeAsImage(tree2, test_output_dir + "/different_tree2.png");

    // Calculate the diff
    auto [updateTree, diffTree] = calcOctreeDiff(tree1, tree2, 0.001);

    // Save the diff trees as images
    saveOctreeAsImage(updateTree, test_output_dir + "/different_update_tree.png");
    saveOctreeAsImage(diffTree, test_output_dir + "/different_diff_tree.png");

    // Check that the differences are correctly identified
    EXPECT_GT(updateTree.size(), 0);
    EXPECT_GT(diffTree.size(), 0);

    // Validate the specific points in the update and diff trees
    auto update_node = updateTree.search(-1.0, -1.0, -1.0);
    ASSERT_NE(update_node, nullptr);
    EXPECT_NEAR(update_node->getLogOdds(), 0.9, 0.1);  // update tree should have log odds of 0.9

    auto diff_node = diffTree.search(-1.0, -1.0, -1.0);
    ASSERT_NE(diff_node, nullptr);
    EXPECT_NEAR(diff_node->getLogOdds(), 0.4, 0.1);  // diff tree should reflect the change in log odds
}

// Main function to run all tests
int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
