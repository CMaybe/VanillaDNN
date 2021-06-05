cmake_minimum_required(VERSION 3.10)

project(mnist)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_CURRENT_LIST_DIR}/bin)
set(CMAKE_BUILD_TYPE Release)
set(THREADS_PREFER_PTHREAD_FLAG ON)

find_package(Threads REQUIRED)
find_package(VanillaDNN PATHS ..)

add_definitions("-std=c++17")

target_link_libraries(mnist PRIVATE Threads::Threads)
target_include_directories(mnist PUBLIC ${VanillaDNN_INCLUDE_DIR})
target_compile_definitions(mnist PRIVATE MNIST_DATA_LOCATION="${MNIST_DATA_DIR}")


add_executable(Main
	source/main.cpp
)