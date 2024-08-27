#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include <rosbag2_cpp/reader.hpp>
#include <rosbag2_storage_default_plugins/sqlite/sqlite_storage.hpp>
#include <rosbag2_storage/serialized_bag_message.hpp>
#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/msg/octomap.hpp>

class OctomapProcessor : public rclcpp::Node {
public:
    OctomapProcessor() : Node("octomap_processor") {}

    int process_bag(const std::string& input_bag_path, const std::string& input_topic) {
        // PROCESS PARAMETERS
        if (input_bag_path.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Input bag path is empty.");
            return 1;
        }
        if (input_topic.empty()) {
            RCLCPP_ERROR(this->get_logger(), "Octomap topic is empty.");
            return 1;
        }

        std::string output_csv_path;
        std::string extension = ".bag";
        if (input_bag_path.size() > extension.size() &&
            input_bag_path.substr(input_bag_path.size() - extension.size()) == extension)
        {
            output_csv_path = input_bag_path.substr(0, input_bag_path.size() - extension.size()) + "_points.csv";
        } else {
            output_csv_path = input_bag_path + "_points.csv";
        }

        // Open the bag file (rosbag2 in ROS 2)
        auto reader = std::make_shared<rosbag2_cpp::Reader>();
        try {
            reader->open(input_bag_path);
        } catch (const std::exception& e) {
            RCLCPP_ERROR(this->get_logger(), "Error opening bag file: %s", e.what());
            return 1;
        }

        // PROCESS DATA
        std::shared_ptr<rosbag2_storage::SerializedBagMessage> last_msg;
        while (reader->has_next()) {
            auto bag_message = reader->read_next();
            if (bag_message->topic_name == input_topic || ("/" + bag_message->topic_name) == input_topic) {
                last_msg = bag_message;
            }
        }

        if (last_msg == nullptr) {
            RCLCPP_ERROR(this->get_logger(), "No messages found on topic '%s'", input_topic.c_str());
            return 1;
        }

        // Deserialize the message
        auto octomap_msg = std::make_shared<octomap_msgs::msg::Octomap>();
        rclcpp::SerializedMessage extracted_serialized_msg(*last_msg->serialized_data);
        rclcpp::Serialization<octomap_msgs::msg::Octomap> serializer;
        serializer.deserialize_message(&extracted_serialized_msg, octomap_msg.get());

        // Convert Octomap message to octomap tree
        octomap::AbstractOcTree* a_tree = octomap_msgs::fullMsgToMap(*octomap_msg);
        octomap::OcTree* tree = dynamic_cast<octomap::OcTree*>(a_tree);
        if (!tree) {
            RCLCPP_ERROR(this->get_logger(), "Failed to create octomap from message");
            delete a_tree;
            return 1;
        }

        // Write leaf data to a CSV file
        std::ofstream output_csv;
        output_csv.open(output_csv_path);
        if (!output_csv.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open CSV file: %s", output_csv_path.c_str());
            return 1;
        }
        output_csv << "x,y,z,occupancy_odds\n";
        for (octomap::OcTree::leaf_iterator it = tree->begin_leafs(), end = tree->end_leafs(); it != end; ++it) {
            output_csv << it.getX() << "," << it.getY() << "," << it.getZ() << "," << it->getLogOdds() << "\n";
        }

        output_csv.close();
        delete a_tree;

        return 0;
    }
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);

    if (argc < 3) {
        std::cerr << "Usage: octomap_processor <input_bag_path> <octomap_topic>" << std::endl;
        return 1;
    }

    auto node = std::make_shared<OctomapProcessor>();
    std::string input_bag_path = argv[1];
    std::string input_topic = argv[2];

    int result = node->process_bag(input_bag_path, input_topic);

    rclcpp::shutdown();
    return result;
}
