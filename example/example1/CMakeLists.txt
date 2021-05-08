cmake_minimum_required(VERSION 3.10)

project(
	Vanilla-DNN-ex1
	VERSION 1.0
	DESCRIPTION "DNN without any lib"
	LANGUAGES CXX
)


add_executable(example1
	source/ex1.cpp
	../../Math/Matrix/Matrix.cpp
	../../Math/Vector/Vector.cpp
	../../DNN/Layer/Layer.cpp
	../../DNN/Model/Model.cpp
	../../DNN/Functions/DNNFunction.hpp
)

target_include_directories(example1 PUBLIC ${CMAKE_SOURCE_DIR}/../..)
