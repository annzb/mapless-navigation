cmake_minimum_required(VERSION 3.5)
project(mapless_navigation)

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(octomap REQUIRED)
find_package(octomap_msgs REQUIRED)
find_package(ament_cmake_gtest REQUIRED)  # For GoogleTest
find_package(Eigen3 REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(rosbag2_storage REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io)

# Manually specify OpenCV paths from pkg-config output
set(OpenCV_INCLUDE_DIRS "/usr/include/opencv4")
set(OpenCV_LIBRARIES "-lopencv_stitching -lopencv_aruco -lopencv_bgsegm -lopencv_bioinspired -lopencv_ccalib -lopencv_dnn_objdetect -lopencv_dnn_superres -lopencv_dpm -lopencv_highgui -lopencv_face -lopencv_freetype -lopencv_fuzzy -lopencv_hdf -lopencv_hfs -lopencv_img_hash -lopencv_line_descriptor -lopencv_quality -lopencv_reg -lopencv_rgbd -lopencv_saliency -lopencv_shape -lopencv_stereo -lopencv_structured_light -lopencv_phase_unwrapping -lopencv_superres -lopencv_optflow -lopencv_surface_matching -lopencv_tracking -lopencv_datasets -lopencv_text -lopencv_dnn -lopencv_plot -lopencv_ml -lopencv_videostab -lopencv_videoio -lopencv_viz -lopencv_ximgproc -lopencv_video -lopencv_xobjdetect -lopencv_objdetect -lopencv_calib3d -lopencv_imgcodecs -lopencv_features2d -lopencv_flann -lopencv_xphoto -lopencv_photo -lopencv_imgproc -lopencv_core")

# Include directories
include_directories(include ${EIGEN3_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS} ${PCL_INCLUDE_DIRS})

# Add library for octree_diff
add_library(octree_diff_lib SHARED src/octree_diff.cpp)
ament_target_dependencies(octree_diff_lib rclcpp octomap octomap_msgs)

# Install the octree_diff_lib shared library
install(TARGETS octree_diff_lib
  LIBRARY DESTINATION lib
)

# Add executable for odom2tf node
add_executable(odom2tf src/odom2tf_node.cpp)
ament_target_dependencies(odom2tf
  rclcpp
  tf2_ros
  geometry_msgs
  nav_msgs
  rosbag2_cpp
  rosbag2_storage
)
target_link_libraries(odom2tf
  rosbag2_cpp::rosbag2_cpp
  rosbag2_storage::rosbag2_storage
)
# Install the octomap_preprocess node executable
install(TARGETS odom2tf
  DESTINATION lib/${PROJECT_NAME}
)

# Add executable for octomap_preprocess node
add_executable(octomap_preprocess src/octomap_preprocess.cpp)
ament_target_dependencies(octomap_preprocess
  rclcpp
  std_msgs
  sensor_msgs
  octomap
  octomap_msgs
  rosbag2_cpp
  rosbag2_storage
)
target_link_libraries(octomap_preprocess
  octree_diff_lib
  ${EIGEN3_LIBRARIES}
  rosbag2_cpp::rosbag2_cpp
  rosbag2_storage::rosbag2_storage
  ${OpenCV_LIBRARIES}
  ${PCL_LIBRARIES}  # Link PCL libraries
)

# Install the octomap_preprocess node executable
install(TARGETS octomap_preprocess
  DESTINATION lib/${PROJECT_NAME}
)

# Add GoogleTest-based test for octree_diff
ament_add_gtest(test_octree_diff test/test_octree_diff.cpp)
target_link_libraries(test_octree_diff octree_diff_lib ${EIGEN3_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES})
ament_target_dependencies(test_octree_diff rclcpp octomap octomap_msgs)

# Install the test executable
install(TARGETS test_octree_diff
  DESTINATION lib/${PROJECT_NAME}
)

install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/
)

# ament package macro
ament_package()
