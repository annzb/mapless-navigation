cmake_minimum_required(VERSION 3.12)
project(coloradar_tools LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-fconcepts)

find_package(PCL 1.12 REQUIRED)
find_package(octomap REQUIRED)
find_package(OpenCV REQUIRED)


include_directories(${PROJECT_SOURCE_DIR}/include ${PCL_INCLUDE_DIRS} ${OCTOMAP_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

### LIBS

add_library(coloradar_tools SHARED libs/utils.cpp libs/octo_pointcloud.cpp libs/coloradar_run.cpp libs/coloradar_dataset.cpp)
# target_include_directories(coloradar_tools PUBLIC ${PROJECT_SOURCE_DIR}/libs)
target_link_libraries(coloradar_tools ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})

# add_library(coloradar_tools SHARED libs/coloradar_tools.h)
# add_library(coloradar_tools SHARED include/utils.hpp include/pcl_functions.hpp include/octo_pointcloud.hpp libs/coloradar_run.cpp libs/coloradar_dataset.cpp)
# target_include_directories(coloradar_tools PRIVATE ${PROJECT_SOURCE_DIR}/libs)
# target_link_libraries(coloradar_tools ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})

# add_library(coloradar_tools SHARED data_prep/coloradar_tools.cpp)
# target_link_libraries(coloradar_tools ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})

# add_library(octree_diff SHARED data_prep/octree_diff.cpp)
# target_link_libraries(octree_diff ${OCTOMAP_LIBRARIES})

### TESTS

add_executable(coloradar_tests test/test_coloradar_tools.cpp)
target_link_libraries(coloradar_tests coloradar_tools ${PCL_LIBRARIES} gtest gtest_main pthread)

# add_executable(octree_diff_tests test/test_octree_diff.cpp)
# target_link_libraries(octree_diff_tests octree_diff ${PCL_LIBRARIES} ${OpenCV_LIBS} gtest gtest_main pthread)


### SCRIPTS

add_executable(filter_cloud data_prep/filter_cloud.cpp)
target_link_libraries(filter_cloud coloradar_tools ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})

add_executable(build_octomap data_prep/build_octomap.cpp)
target_link_libraries(build_octomap coloradar_tools ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})


# add_definitions(${PCL_DEFINITIONS})
