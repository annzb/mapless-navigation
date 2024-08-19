cmake_minimum_required(VERSION 3.5)
project(mapless_navigation)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(std_msgs REQUIRED)
find_package(octomap_ros REQUIRED)
find_package(octomap_msgs REQUIRED)

# Include directories
include_directories(include)

# Executable and link libraries
add_executable(octomap_processor src/octomap_postprocess.cpp)
ament_target_dependencies(octomap_processor rclcpp std_msgs octomap_ros octomap_msgs)

# Install executable
install(TARGETS octomap_processor
  DESTINATION lib/${PROJECT_NAME})

# ament package macro
ament_package()

