cmake_minimum_required(VERSION 3.12)
project(coloradar_tools)


find_program(GCC_COMPILER NAMES gcc-12 gcc-11 gcc-10 gcc-9 gcc-8 gcc-7 PATHS /usr/bin NO_DEFAULT_PATH)
if (NOT GCC_COMPILER)
    message(FATAL_ERROR "No suitable GCC compiler found below version 13!")
endif()
string(REGEX REPLACE "gcc-[0-9]+" "g++" GPP_COMPILER ${GCC_COMPILER})
set(CMAKE_C_COMPILER ${GCC_COMPILER} CACHE PATH "C compiler" FORCE)
set(CMAKE_CXX_COMPILER ${GPP_COMPILER} CACHE PATH "C++ compiler" FORCE)
message(STATUS "Using compiler: ${GCC_COMPILER}")


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_options(-fconcepts)


find_package(MPI REQUIRED)
find_package(VTK COMPONENTS mpi REQUIRED)
find_package(PCL 1.12 REQUIRED)
find_package(octomap REQUIRED)
find_package(OpenCV REQUIRED)
find_package(CUDA 12.6)

set(CUDA_LIBS ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})
include_directories(${PROJECT_SOURCE_DIR}/include ${PCL_INCLUDE_DIRS} ${OCTOMAP_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})

### LIBS

add_library(coloradar_tools SHARED
    libs/utils.cpp
    libs/configs.cpp
    libs/pcl_functions.cpp
    libs/octo_pointcloud.cpp
    libs/coloradar_run.cpp
    libs/coloradar_dataset.cpp
)
target_link_libraries(coloradar_tools ${PCL_LIBRARIES} ${OCTOMAP_LIBRARIES})


if(CUDA_FOUND)
    message(STATUS "CUDA found, compiling CUDA code.")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -std=c++20")

    cuda_add_library(coloradar_cuda SHARED
        libs/cuda_kernels.cu
        libs/radar_processing.cu
    )
    target_link_libraries(coloradar_cuda coloradar_tools ${CUDA_LIBS})
    include_directories(${CUDA_INCLUDE_DIRS})

    cuda_add_executable(compare_heatmaps data_prep/compare_heatmaps.cu)
    target_link_libraries(compare_heatmaps coloradar_cuda)
else()
    message(STATUS "CUDA not found, skipping CUDA compilation.")
endif()


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

add_executable(create_cascade_pointclouds data_prep/create_cascade_pointclouds.cpp)
target_link_libraries(create_cascade_pointclouds coloradar_tools)
