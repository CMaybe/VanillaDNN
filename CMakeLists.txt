cmake_minimum_required(VERSION 3.10)

project(
	VanillaDNN
	VERSION 1.0
	DESCRIPTION "DNN without any lib"
	LANGUAGES CXX
)

set(MNIST_DATA_DIR ${CMAKE_CURRENT_LIST_DIR})
add_definitions("-std=c++17")

add_executable(Main
	src/main.cpp
	Math/Matrix/Matrix.cpp
	Math/Vector/Vector.cpp
	DNN/Functions/DNNFunction.cpp
	DNN/Layer/Layer.cpp
	DNN/Model/Model.cpp
	MNIST/MNIST.cpp
)

target_include_directories(Main PUBLIC ${CMAKE_SOURCE_DIR})
