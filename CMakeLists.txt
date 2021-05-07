cmake_minimum_required(VERSION 3.10)

project(
	Vanilla-DNN
	VERSION 1.0
	DESCRIPTION "DNN without any lib"
	LANGUAGES CXX
)


add_executable(Main
	src/main.cpp
	Math/Matrix/Matrix.cpp
	Math/Vector/Vector.cpp
	DNN/Layer/Layer.cpp
	DNN/Model/Model.cpp
	DNN/Functions/DNNFunction.hpp
)

target_include_directories(Main PUBLIC ${CMAKE_SOURCE_DIR})
