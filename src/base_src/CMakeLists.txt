cmake_minimum_required(VERSION 3.10)
project(build_octomap)

set(CMAKE_CXX_STANDARD 17)

find_package(PCL REQUIRED)
find_package(octomap REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include ${PCL_INCLUDE_DIRS} ${OCTOMAP_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})


### LIBS
add_library(coloradar_tools SHARED data_prep/coloradar_tools.cpp)
target_link_libraries(coloradar_tools ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})

add_library(octree_diff SHARED data_prep/octree_diff.cpp)
target_link_libraries(octree_diff ${OCTOMAP_LIBRARIES})


### SCRIPTS
add_executable(filter_cloud data_prep/filter_cloud.cpp)
target_link_libraries(filter_cloud coloradar_tools ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})

add_executable(build_octomap data_prep/build_octomap.cpp)
target_link_libraries(build_octomap coloradar_tools octree_diff ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})


### TESTS
add_executable(octree_diff_tests test/test_octree_diff.cpp)
target_link_libraries(octree_diff_tests octree_diff ${PCL_LIBRARIES} ${OpenCV_LIBS} gtest gtest_main pthread)

add_executable(coloradar_tests test/test_coloradar_tools.cpp)
target_link_libraries(coloradar_tests coloradar_tools ${PCL_LIBRARIES} gtest gtest_main pthread)

add_definitions(${PCL_DEFINITIONS})
