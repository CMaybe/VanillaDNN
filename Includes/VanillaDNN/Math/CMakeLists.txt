cmake_minimum_required(VERSION 3.10)

project(Math)
add_definitions("-std=c++17")


set(CMAKE_CXX_FLAGS "-O3")
set(CMAKE_BUILD_TYPE Release)
add_library(Math STATIC
	./Matrix/Matrix.cpp
	./Vector/Vector.hpp
)
