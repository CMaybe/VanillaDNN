cmake_minimum_required(VERSION 3.10)

project(mnist)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-O3")
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR}/bin/${CMAKE_BUILD_TYPE})
find_package(VanillaDNN PATHS ${CMAKE_CURRENT_LIST_DIR}/../../../  REQUIRED )
add_definitions("-std=c++17")

add_executable(mnist
	source/mnist.cpp
)
target_link_libraries(mnist ${VanillaDNN_LIBRARY_DIR}/${CMAKE_BUILD_TYPE}/libVanillaDNN.a)
target_compile_definitions(mnist PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")

