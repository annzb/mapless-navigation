cmake_minimum_required(VERSION 3.0.2)
project(mapless-navigation)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rosbag
  std_msgs
  octomap_ros
  octomap_msgs
)

catkin_package(
  CATKIN_DEPENDS roscpp rosbag std_msgs octomap_ros octomap_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

add_executable(octomap_processor src/octomap_postprocess.cpp)
target_link_libraries(octomap_processor ${catkin_LIBRARIES})
add_dependencies(octomap_processor ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})
