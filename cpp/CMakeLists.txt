cmake_minimum_required(VERSION 3.20)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# 这一行必须在target_link_libraries之前
add_executable(cpp src/main.cpp
        src/inference.hpp
        src/detect.hpp)

# 将库链接到项目中
target_link_libraries(cpp ${OpenCV_LIBS})