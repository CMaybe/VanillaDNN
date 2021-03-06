cmake_minimum_required(VERSION 3.10)

project(VanillaDNN)
find_package(Threads REQUIRED)
add_definitions("-std=c++17")


set(VanillaDNN_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/Includes)
set(VanillaDNN_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/Sources)
set(CMAKE_CXX_FLAGS "-O3")
set(THREADS_PREFER_PTHREAD_FLAG ON)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib/${CMAKE_BUILD_TYPE})

add_library(VanillaDNN STATIC
	${VanillaDNN_INCLUDE_DIR}/VanillaDNN/Math/Vector/Vector.cpp
	${VanillaDNN_INCLUDE_DIR}/VanillaDNN/Math/Matrix/Matrix.cpp
	${VanillaDNN_SOURCE_DIR}/VanillaDNN/Functions/Activation.cpp
	${VanillaDNN_SOURCE_DIR}/VanillaDNN/Functions/Loss.cpp
	${VanillaDNN_SOURCE_DIR}/VanillaDNN/Functions/Optimizer.cpp
	${VanillaDNN_SOURCE_DIR}/VanillaDNN/Layers/DenseLayer.cpp
	${VanillaDNN_SOURCE_DIR}/VanillaDNN/Model/Model.cpp
	${VanillaDNN_SOURCE_DIR}/VanillaDNN/MNIST/MNIST.cpp
)

target_link_libraries(VanillaDNN PRIVATE Threads::Threads)
target_include_directories(VanillaDNN PUBLIC ${VanillaDNN_INCLUDE_DIR})