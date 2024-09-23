cmake_minimum_required(VERSION 3.12)
project(coloradar_tools)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-fconcepts)

find_package(MPI REQUIRED)
find_package(VTK COMPONENTS mpi REQUIRED)
find_package(PCL 1.12 REQUIRED)
find_package(octomap REQUIRED)
find_package(OpenCV REQUIRED)


include_directories(${PROJECT_SOURCE_DIR}/include ${PCL_INCLUDE_DIRS} ${OCTOMAP_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

### LIBS

add_library(coloradar_tools SHARED libs/utils.cpp libs/octo_pointcloud.cpp libs/coloradar_run.cpp libs/coloradar_dataset.cpp)
target_link_libraries(coloradar_tools ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})


### TESTS

add_executable(coloradar_tests test/test_coloradar_tools.cpp)
target_link_libraries(coloradar_tests coloradar_tools gtest gtest_main pthread)


### SCRIPTS

add_executable(interpolate_poses_for_lidar data_prep/interpolate_poses_for_lidar.cpp)
target_link_libraries(interpolate_poses_for_lidar coloradar_tools)

add_executable(filter_cloud data_prep/filter_cloud.cpp)
target_link_libraries(filter_cloud coloradar_tools)

add_executable(build_octomap data_prep/build_octomap.cpp)
target_link_libraries(build_octomap coloradar_tools)

add_executable(sample_map_frames data_prep/sample_map_frames.cpp)
target_link_libraries(sample_map_frames coloradar_tools)


# add_definitions(${PCL_DEFINITIONS})
