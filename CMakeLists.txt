cmake_minimum_required(VERSION 3.10)

# set the project name
project(Project)

# specify the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "-O3 -DNDEBUG")

# add include directory
include_directories(include)

# add libraries
find_package(Eigen3 REQUIRED)

# add the executable
add_executable(Project src/main.cpp)

target_link_libraries(Project Eigen3::Eigen)