cmake_minimum_required(VERSION 3.5)

project(Reconstruction VERSION 1.0)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++14")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include_directories("/usr/local/include/eigen3")

find_package(Sophus REQUIRED)
include_directories(${Sophus_INCLUDE_DIRS})
find_package(fmt REQUIRED)
include_directories(${fmt_INCLUDE_DIRS})

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
#include_directories("/home/pinhao/pinhao/opencv/include/opencv2")

find_package(PCL 1.12 REQUIRED COMPONENTS common io)
find_package(PCL 1.12 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})

add_definitions(${PCL_DEFINITIONS})

set(CUR_DIR "/home/pinhao/RGBD_SLAM")
include_directories(${CUR_DIR})

add_library(utils src/utils.cpp include/utils.h)
target_include_directories(utils PRIVATE ${CUR_DIR})
add_library(feature_helper src/feature_helper.cpp include/feature_helper.h)
target_include_directories(feature_helper PRIVATE ${CUR_DIR})

add_library(rgbd src/rgbd.cpp include/rgbd.h)
target_include_directories(rgbd PRIVATE ${CUR_DIR})
add_library(slam src/slam.cpp include/slam.h)
target_include_directories(slam PRIVATE ${CUR_DIR})


add_executable(output src/main.cpp include/slam.h include/feature_helper.h include/rgbd.h include/utils.h)


target_link_libraries(rgbd ${OpenCV_LIBS})
target_link_libraries(rgbd utils)


target_link_libraries(slam ${PCL_LIBRARIES})
target_link_libraries(slam ${OpenCV_LIBS})
target_link_libraries(slam rgbd)
target_link_libraries(slam utils)
target_link_libraries(slam feature_helper)

target_link_libraries(output rgbd)
target_link_libraries(output slam)
target_link_libraries(output fmt)
