cmake_minimum_required(VERSION 3.10)

project(
	VanillaDNN
	VERSION 1.0
	DESCRIPTION "DNN without any lib"
	LANGUAGES CXX
)

set(CMAKE_BUILD_TYPE Release)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

add_definitions("-std=c++17")

add_executable(Main
	src/main.cpp
	Math/Matrix/Matrix.cpp
	Math/Vector/Vector.cpp
	DNN/Functions/DNNFunction.cpp
	DNN/Functions/Optimizer.cpp	
	DNN/Layer/Layer.cpp
	DNN/Model/Model.cpp
	MNIST/MNIST.cpp
)

target_link_libraries(Main PRIVATE Threads::Threads)
target_include_directories(Main PUBLIC ${CMAKE_SOURCE_DIR})
