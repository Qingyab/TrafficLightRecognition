cmake_minimum_required(VERSION 3.10)
link_directories(C:/Tools_/A_cmputer/opencv/opencv-4.5.5/build)
set(CMAKE_PREFIX_PATH "C:/Tools_/A_cmputer/opencv/opencv-4.5.5/build/install")

project(TrafficLightRecognition)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARY_DIRS})
add_definitions(${OpenCV_DEFINITIONS})


file(GLOB SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
add_executable(${PROJECT_NAME} ${SRC})

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
set(GLOB L "C:/Tools_/A_cmputer/opencv/opencv-4.5.5/build/bin")
target_link_libraries(${PROJECT_NAME} ${L})
