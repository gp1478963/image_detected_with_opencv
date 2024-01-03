cmake_minimum_required(VERSION 3.23)
project(untitled1)

set(CMAKE_CXX_STANDARD 20)

set(OpenCV_DIR "D:\\dev\\opencv\\build\\x64\\vc16\\lib")
find_package(OpenCV REQUIRED)
link_directories("D:\\dev\\opencv\\build\\x64\\vc16\\bin")
message(STATUS ${OpenCV_LIBS})
add_executable(untitled1 main.cpp augment.cpp)
target_link_libraries(untitled1 PRIVATE ${OpenCV_LIBS})
target_include_directories(untitled1 PRIVATE ${OpenCV_INCLUDE_DIRS})
