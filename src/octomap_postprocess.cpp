#include <fstream>

#include <ros/ros.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <boost/foreach.hpp>

#include <octomap/octomap.h>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/Octomap.h>


int main(int argc, char** argv) {
// PROCESS PARAMETERS
    if (argc < 3) {
        ROS_ERROR("Usage: octomap_processor <input_bag_path> <octomap_topic>");
        return 1;
    }
    std::string input_bag_path = argv[1];
    std::string input_topic = argv[2];
    if (input_bag_path.empty()) {
        ROS_ERROR("Input bag path is empty.");
        return 1;
    }
    if (input_topic.empty()) {
        ROS_ERROR("Octomap topic is empty.");
        return 1;
    }

// PROCESS INPUT BAG
    std::string output_csv_path;
    std::string extension = ".bag";
    if (input_bag_path.size() > extension.size() &&
        input_bag_path.substr(input_bag_path.size() - extension.size()) == extension)
    {
        output_csv_path = input_bag_path.substr(0, input_bag_path.size() - extension.size()) + "_points.csv";
    } else {
        output_csv_path = input_bag_path + "_points.csv";
    }

    rosbag::Bag input_bag;
    try {
        input_bag.open(input_bag_path, rosbag::bagmode::Read);
    } catch(rosbag::BagException& e) {
        ROS_ERROR("Error opening bag file: %s", e.what());
        return 1;
    }
    rosbag::View view(input_bag, rosbag::TopicQuery({input_topic}));

    // check if required topics exist
    bool input_topic_found = false;
    BOOST_FOREACH(const rosbag::ConnectionInfo *info, view.getConnections()) {
      if (info->topic == input_topic) {
        input_topic_found = true;
      }
    }
    if (!input_topic_found) {
        ROS_ERROR("Required topic '%s' not found in file '%s'", input_topic.c_str(), input_bag_path.c_str());
        return 1;
    }
    
// PROCESS DATA
    // Find and process the last message in the topic
    octomap_msgs::Octomap::ConstPtr last_msg;
    for (const rosbag::MessageInstance& m : view) {
        if (m.getTopic() == input_topic || ("/" + m.getTopic()) == input_topic) {
            last_msg = m.instantiate<octomap_msgs::Octomap>();
        }
    }
    if (last_msg == nullptr) {
        ROS_ERROR("No messages found on topic '%s'", input_topic.c_str());
        return 1;
    }
    octomap::AbstractOcTree* a_tree = octomap_msgs::fullMsgToMap(*last_msg);
    octomap::OcTree* tree = dynamic_cast<octomap::OcTree*>(a_tree);
    if (!tree) {
        ROS_ERROR("Failed to create octomap from message");
        delete a_tree;
        return 1;
    }

    // Write leaf data to a CSV file
    std::ofstream output_csv;
    output_csv.open(output_csv_path);
    if (!output_csv.is_open()) {
        ROS_ERROR("Failed to open CSV file: %s", output_csv_path.c_str());
        return 1;
    }
    output_csv << "x,y,z,occupancy_odds\n";
    for (octomap::OcTree::leaf_iterator it = tree->begin_leafs(), end = tree->end_leafs(); it != end; ++it) {
        output_csv << it.getX() << "," << it.getY() << "," << it.getZ() << "," << it->getLogOdds() << "\n";
    }

    output_csv.close();
    delete a_tree;
    input_bag.close();

    return 0;
}
