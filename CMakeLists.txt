cmake_minimum_required(VERSION 3.9)
project(shape_detection)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

find_package( OpenCV REQUIRED )
set(SOURCE_FILES main.cpp)
add_executable(shape_detection ${SOURCE_FILES})
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries( shape_detection ${OpenCV_LIBS} )

configure_file(shapes.png shapes.png COPYONLY)