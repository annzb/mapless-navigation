cmake_minimum_required(VERSION 3.10)
project(build_octomap)

set(CMAKE_CXX_STANDARD 17)

find_package(PCL REQUIRED)
find_package(octomap REQUIRED)
find_package(OpenCV REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include ${PCL_INCLUDE_DIRS} ${OCTOMAP_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

add_library(coloradar_tools SHARED data_prep/coloradar_tools.cpp)
target_link_libraries(coloradar_tools ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})

add_library(octree_diff SHARED data_prep/octree_diff.cpp)
target_link_libraries(octree_diff ${OCTOMAP_LIBRARIES})

add_executable(build_octomap data_prep/build_octomap.cpp)
target_link_libraries(build_octomap coloradar_tools octree_diff ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})

add_executable(tests test/test_octree_diff.cpp)
target_link_libraries(tests octree_diff ${PCL_LIBRARIES} ${OpenCV_LIBS} gtest gtest_main pthread)

add_definitions(${PCL_DEFINITIONS})
