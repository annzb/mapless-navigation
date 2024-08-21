#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <octomap/octomap.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>
#include <eigen3/Eigen/Geometry>
#include <tf2_eigen/tf2_eigen.h>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>
#include <rosbag2_storage_default_plugins/sqlite/sqlite_storage.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <std_msgs/msg/float64_multi_array.hpp>
#include "octree_diff.h"

// Helper function to convert pose message to Eigen
Eigen::Affine3d poseMsgToEigen(const nav_msgs::msg::Odometry::SharedPtr &msg)
{
    geometry_msgs::msg::Pose pose = msg->pose.pose;
    Eigen::Affine3d transform = Eigen::Translation3d(pose.position.x, pose.position.y, pose.position.z) *
                                Eigen::Quaterniond(pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z);
    return transform.inverse();
}

// Helper function to convert pcl::PointCloud to sensor_msgs::msg::PointCloud2
sensor_msgs::msg::PointCloud2 pclToMsg(pcl::PointCloud<pcl::PointXYZ> &cloud, const std::string &frame_id, const rclcpp::Time &stamp)
{
    sensor_msgs::msg::PointCloud2 cloudMsg;
    pcl::toROSMsg(cloud, cloudMsg);
    cloudMsg.header.frame_id = frame_id;
    cloudMsg.header.stamp = stamp;
    return cloudMsg;
}

// Helper function to convert octomap to PointCloud and occupancy array
std::tuple<std_msgs::msg::Float64MultiArray, pcl::PointCloud<pcl::PointXYZ>, pcl::PointCloud<pcl::PointXYZ>, pcl::PointCloud<pcl::PointXYZ>>
octreeToPointCloud(const octomap::OcTree &tree, const std::string &frame_id, const rclcpp::Time &stamp, const Eigen::Affine3d &center)
{
    pcl::PointCloud<pcl::PointXYZ> pclUnoccupied, pclCloudCentered, pclCloud;
    std_msgs::msg::Float64MultiArray occupancyOdds;
    occupancyOdds.data.clear();

    for (octomap::OcTree::leaf_iterator it = tree.begin_leafs(), end = tree.end_leafs(); it != end; ++it)
    {
        Eigen::Vector3d pointOrig(it.getX(), it.getY(), it.getZ());
        Eigen::Vector3d pointCentered = center * pointOrig;

        pcl::PointXYZ pointOrigPCL(pointOrig.x(), pointOrig.y(), pointOrig.z());
        pcl::PointXYZ pointCenteredPCL(pointCentered.x(), pointCentered.y(), pointCentered.z());

        pclUnoccupied.points.push_back(pointCenteredPCL);
        occupancyOdds.data.push_back(it->getLogOdds());

        if (tree.isNodeOccupied(*it))
        {
            pclCloud.points.push_back(pointOrigPCL);
            pclCloudCentered.points.push_back(pointCenteredPCL);
        }
    }
    return std::make_tuple(occupancyOdds, pclUnoccupied, pclCloudCentered, pclCloud);
}

class OctomapPostProcessNode : public rclcpp::Node
{
public:
    OctomapPostProcessNode()
        : Node("octomap_postprocess_node")
    {
        // Declare parameters
        input_bag_path_ = this->declare_parameter<std::string>("input_bag_path", "");
        input_topic_ = this->declare_parameter<std::string>("input_topic", "");
        odometry_topic_ = this->declare_parameter<std::string>("odometry_topic", "");
        marker_lifetime_ = this->declare_parameter<double>("marker_lifetime", 0.1);

        if (input_bag_path_.empty() || input_topic_.empty() || odometry_topic_.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Parameters input_bag_path, input_topic, and odometry_topic must be provided.");
            return;
        }

        setup_bag_paths();
        process_bags();
    }

private:
    std::string input_bag_path_;
    std::string output_bag_path_;
    std::string input_topic_;
    std::string odometry_topic_;
    double marker_lifetime_;

    // Set up input/output bag paths
    void setup_bag_paths()
    {
        std::string extension = ".bag";
        if (input_bag_path_.size() > extension.size() &&
            input_bag_path_.substr(input_bag_path_.size() - extension.size()) == extension)
        {
            output_bag_path_ = input_bag_path_.substr(0, input_bag_path_.size() - extension.size()) + "_diff.bag";
        }
        else
        {
            output_bag_path_ = input_bag_path_ + "_diff.bag";
        }
    }

    // Process bags using rosbag2
    void process_bags()
    {
        try
        {
            rosbag2_cpp::Reader input_bag;
            rosbag2_cpp::Writer output_bag;
            input_bag.open(input_bag_path_);
            output_bag.open(output_bag_path_);

            // Initialize variables
            Eigen::Affine3d odometry_position = Eigen::Affine3d::Identity();
            octomap::OcTree *prev_tree = nullptr;

            // Process messages
            while (input_bag.has_next())
            {
                auto msg = input_bag.read_next();
                auto topic_name = msg->topic_name;

                if (topic_name == odometry_topic_)
                {
                    auto o_msg = std::make_shared<nav_msgs::msg::Odometry>();
                    // Deserialize the odometry message here
                    // Apply the odometry pose to center the point clouds
                    odometry_position = poseMsgToEigen(o_msg);
                }
                else if (topic_name == input_topic_)
                {
                    auto i_msg = std::make_shared<octomap_msgs::msg::Octomap>();
                    // Deserialize the octomap message here
                    octomap::OcTree *current_tree = dynamic_cast<octomap::OcTree *>(octomap_msgs::msgToMap(*i_msg));
                    if (prev_tree != nullptr)
                    {
                        std::string frame_id = i_msg->header.frame_id;
                        rclcpp::Time stamp = i_msg->header.stamp;

                        // Calculate tree diff
                        std::pair<octomap::OcTree, octomap::OcTree> diff_trees = calcOctreeDiff(*prev_tree, *current_tree);

                        // Convert to PointCloud and occupancy arrays
                        auto [updateOccupancyOdds, updatePclUnoccupied, updatePclCentered, updatePcl] =
                            octreeToPointCloud(diff_trees.first, frame_id, stamp, odometry_position);
                        auto [diffOccupancyOdds, diffPclUnoccupied, diffPclCentered, diffPcl] =
                            octreeToPointCloud(diff_trees.second, frame_id, stamp, odometry_position);

                        // Convert PointClouds to ROS messages
                        sensor_msgs::msg::PointCloud2 updatePclUnoccupiedMsg = pclToMsg(updatePclUnoccupied, frame_id, stamp);
                        sensor_msgs::msg::PointCloud2 updatePclCenteredMsg = pclToMsg(updatePclCentered, frame_id, stamp);
                        sensor_msgs::msg::PointCloud2 diffPclUnoccupiedMsg = pclToMsg(diffPclUnoccupied, frame_id, stamp);
                        sensor_msgs::msg::PointCloud2 diffPclCenteredMsg = pclToMsg(diffPclCentered, frame_id, stamp);

                        // Write the diff trees and PointClouds to the output bag using templated write()
                        output_bag.write<sensor_msgs::msg::PointCloud2>(updatePclCenteredMsg, input_topic_ + "/update/pcl_centered", stamp);
                        output_bag.write<sensor_msgs::msg::PointCloud2>(updatePclUnoccupiedMsg, input_topic_ + "/update/pcl_unoccupied", stamp);
                        output_bag.write<sensor_msgs::msg::PointCloud2>(diffPclCenteredMsg, input_topic_ + "/diff/pcl_centered", stamp);
                        output_bag.write<sensor_msgs::msg::PointCloud2>(diffPclUnoccupiedMsg, input_topic_ + "/diff/pcl_unoccupied", stamp);

                        // Write the Octomap diff trees to the output bag
                        octomap_msgs::msg::Octomap diff_tree_msg, update_tree_msg;
                        octomap_msgs::binaryMapToMsg(diff_trees.first, update_tree_msg);
                        octomap_msgs::binaryMapToMsg(diff_trees.second, diff_tree_msg);

                        diff_tree_msg.header.frame_id = frame_id;
                        diff_tree_msg.header.stamp = stamp;
                        update_tree_msg.header.frame_id = frame_id;
                        update_tree_msg.header.stamp = stamp;

                        output_bag.write<octomap_msgs::msg::Octomap>(update_tree_msg, input_topic_ + "/update/octomap", stamp);
                        output_bag.write<octomap_msgs::msg::Octomap>(diff_tree_msg, input_topic_ + "/diff/octomap", stamp);

                        // Clean up
                        delete prev_tree;
                    }

                    prev_tree = current_tree;
                }
            }

            if (prev_tree != nullptr)
            {
                delete prev_tree;
            }

            input_bag.close();
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(this->get_logger(), "Error processing bags: %s", e.what());
        }
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OctomapPostProcessNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}