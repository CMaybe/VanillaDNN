cmake_minimum_required(VERSION 3.10)

project(mnist)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR}/bin)
set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Degug)


find_package(VanillaDNN REQUIRED)
add_definitions("-std=c++17")

add_executable(mnist
	source/mnist.cpp
)
target_link_libraries(mnist ${VanillaDNN_LIBRARY_DIR}/libVanillaDNN.a)
target_compile_definitions(mnist PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

