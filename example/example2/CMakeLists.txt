cmake_minimum_required(VERSION 3.10)

project(
	VanillaDNN-ex2
	VERSION 1.0
	DESCRIPTION "DNN without any lib"
	LANGUAGES CXX
)

set(MNIST_DATA_DIR ${CMAKE_CURRENT_LIST_DIR})

add_definitions("-std=c++17")
add_executable(example2
	source/ex2.cpp
	../../Math/Matrix/Matrix.cpp
	../../Math/Vector/Vector.cpp
	../../MNIST/MNIST.cpp
)

target_include_directories(example2 PUBLIC ${CMAKE_SOURCE_DIR}/../..)
