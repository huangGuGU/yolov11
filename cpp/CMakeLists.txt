cmake_minimum_required(VERSION 3.20)
project(cpp)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(cpp src/main.cpp
        src/inference.cpp
        src/detect.cpp
        src/preprocess.cpp
        src/Common.hpp)


target_link_libraries(cpp ${OpenCV_LIBS})