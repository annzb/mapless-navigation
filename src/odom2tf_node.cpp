#include <rclcpp/rclcpp.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <rclcpp/qos.hpp>

class OdomToTFNode : public rclcpp::Node
{
public:
    OdomToTFNode()
        : Node("odom2tf")
    {
        // Customize QoS settings to ensure compatibility
        rclcpp::QoS qos_settings = rclcpp::QoS(rclcpp::QoSInitialization::from_rmw(rmw_qos_profile_default)).reliable().durability_volatile();

        // Create a subscription to the /lidar_ground_truth topic with compatible QoS settings
        sub_odom_ = this->create_subscription<nav_msgs::msg::Odometry>(
            "/lidar_ground_truth",
            qos_settings,
            std::bind(&OdomToTFNode::odomCallback, this, std::placeholders::_1));

        // Create the transform broadcaster
        tf_broadcaster_ = std::make_shared<tf2_ros::TransformBroadcaster>(this);
    }

private:
    void odomCallback(const nav_msgs::msg::Odometry::SharedPtr odom_msg)
    {
        // Convert the odometry pose to a TF transform
        geometry_msgs::msg::TransformStamped transformStamped;
        transformStamped.header.stamp = odom_msg->header.stamp;
        transformStamped.header.frame_id = odom_msg->header.frame_id;
        transformStamped.child_frame_id = "imu_viz_link";

        transformStamped.transform.translation.x = odom_msg->pose.pose.position.x;
        transformStamped.transform.translation.y = odom_msg->pose.pose.position.y;
        transformStamped.transform.translation.z = odom_msg->pose.pose.position.z;
        transformStamped.transform.rotation = odom_msg->pose.pose.orientation;

        // Broadcast the transform
        tf_broadcaster_->sendTransform(transformStamped);
    }

    rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;
    std::shared_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<OdomToTFNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
