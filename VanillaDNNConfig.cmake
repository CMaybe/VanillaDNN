cmake_minimum_required(VERSION 3.10)



find_package(Threads REQUIRED)

set(THREADS_PREFER_PTHREAD_FLAG ON)
set(CMAKE_CXX_FLAGS "-O3")
set(MNIST_DATA_DIR ${CMAKE_CURRENT_LIST_DIR}/Includes/VanillaDNN/MNIST/data)
set(VanillaDNN_INCLUDE_DIR ${CMAKE_CURRENT_LIST_DIR}/Includes)
set(VanillaDNN_SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/Sources)
set(VanillaDNN_LIBRARY_DIR ${CMAKE_CURRENT_LIST_DIR}/lib)

link_libraries(Threads::Threads)
link_libraries(${VanillaDNN_LIBRARY_DIR}/libVanillaDNN.a)
include_directories(${VanillaDNN_INCLUDE_DIR})
