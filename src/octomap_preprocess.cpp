#include <rclcpp/rclcpp.hpp>
#include <rclcpp/duration.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <octomap/octomap.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <octomap_msgs/conversions.h>
#include <eigen3/Eigen/Geometry>
#include <tf2_eigen/tf2_eigen.h>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_cpp/writer.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>
#include <rosbag2_storage_default_plugins/sqlite/sqlite_storage.hpp>
#include <rclcpp/serialization.hpp>

#include "octree_diff.h"

class OctomapPostProcessNode : public rclcpp::Node
{
public:
    OctomapPostProcessNode()
        : Node("octomap_postprocess_node")
    {
        // Declare and get parameters
        input_bag_path_ = this->declare_parameter<std::string>("input_bag_path", "");
        input_topic_ = this->declare_parameter<std::string>("input_topic", "");
        odometry_topic_ = this->declare_parameter<std::string>("odometry_topic", "");
        marker_lifetime_ = this->declare_parameter<double>("marker_lifetime", 0.1);
        voxel_radius_ = this->declare_parameter<double>("voxel_radius", 0);
        target_N_points_ = this->declare_parameter<int>("target_N_points", 0);

        // Parameter validation
        if (input_bag_path_.empty() || input_topic_.empty() || odometry_topic_.empty())
        {
            RCLCPP_ERROR(this->get_logger(), "Input bag path, input topic, and odometry topic must be provided");
            return;
        }

        // Setup input/output bag paths
        setup_bag_paths();

        // Process bag files
        process_bags();
    }

private:
    std::string input_bag_path_;
    std::string output_bag_path_;
    std::string input_topic_;
    std::string odometry_topic_;
    double marker_lifetime_;
    double voxel_radius_;
    int target_N_points_;

    // Convert pose message to Eigen
    Eigen::Affine3d poseMsgToEigen(const nav_msgs::msg::Odometry::SharedPtr &msg)
    {
        geometry_msgs::msg::Pose pose = msg->pose.pose;
        geometry_msgs::msg::Transform transform;
        transform.translation.x = pose.position.x;
        transform.translation.y = pose.position.y;
        transform.translation.z = pose.position.z;
        transform.rotation = pose.orientation;
        Eigen::Affine3d eigenTransform = tf2::transformToEigen(transform);
        return eigenTransform.inverse();
    }

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

            // Serializer for Octomap messages
            rclcpp::Serialization<octomap_msgs::msg::Octomap> serializer;

            // Process messages
            Eigen::Affine3d odometry_position = Eigen::Affine3d::Identity();
            octomap::OcTree *prev_tree = nullptr;

            while (input_bag.has_next())
            {
                auto msg = input_bag.read_next();
                auto topic_name = msg->topic_name;

                if (topic_name == odometry_topic_)
                {
                    auto o_msg = std::make_shared<nav_msgs::msg::Odometry>();
                    // Deserialize the odometry message here (you'll need to actually deserialize the message)
                    odometry_position = poseMsgToEigen(o_msg);
                }
                else if (topic_name == input_topic_)
                {
                    auto i_msg = std::make_shared<octomap_msgs::msg::Octomap>();
                    // Deserialize the octomap message here (you'll need to actually deserialize the message)
                    octomap::OcTree *current_tree = dynamic_cast<octomap::OcTree *>(octomap_msgs::msgToMap(*i_msg));
                    if (current_tree)
                    {
                        if (prev_tree != nullptr)
                        {
                            // Calculate tree diff
                            std::pair<octomap::OcTree, octomap::OcTree> diff_trees = calcOctreeDiff(*prev_tree, *current_tree);

                            // Convert to messages
                            octomap_msgs::msg::Octomap diff_tree_msg, update_tree_msg;
                            octomap_msgs::binaryMapToMsg(diff_trees.first, update_tree_msg);
                            octomap_msgs::binaryMapToMsg(diff_trees.second, diff_tree_msg);

                            // Add headers
                            diff_tree_msg.header.frame_id = i_msg->header.frame_id;
                            diff_tree_msg.header.stamp = i_msg->header.stamp;
                            update_tree_msg.header.frame_id = i_msg->header.frame_id;
                            update_tree_msg.header.stamp = i_msg->header.stamp;

                            // Serialize and write the diff_tree_msg
                            rclcpp::SerializedMessage serialized_diff;
                            serializer.serialize_message(&diff_tree_msg, &serialized_diff);

                            auto serialized_diff_msg = std::make_shared<rosbag2_storage::SerializedBagMessage>();
                            serialized_diff_msg->topic_name = input_topic_ + "/diff";
                            serialized_diff_msg->time_stamp = msg->time_stamp;
                            serialized_diff_msg->serialized_data = std::make_shared<rcutils_uint8_array_t>(
                                serialized_diff.get_rcl_serialized_message());

                            output_bag.write(serialized_diff_msg);

                            // Serialize and write the update_tree_msg
                            rclcpp::SerializedMessage serialized_update;
                            serializer.serialize_message(&update_tree_msg, &serialized_update);

                            auto serialized_update_msg = std::make_shared<rosbag2_storage::SerializedBagMessage>();
                            serialized_update_msg->topic_name = input_topic_ + "/update";
                            serialized_update_msg->time_stamp = msg->time_stamp;
                            serialized_update_msg->serialized_data = std::make_shared<rcutils_uint8_array_t>(
                                serialized_update.get_rcl_serialized_message());

                            output_bag.write(serialized_update_msg);

                            // Clean up
                            delete prev_tree;
                        }

                        prev_tree = current_tree;
                    }
                }
            }

            if (prev_tree != nullptr)
            {
                delete prev_tree;
            }

            // No need to explicitly close the output_bag; it will be closed automatically when it goes out of scope.
            input_bag.close();  // Closing the input bag is still required
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

