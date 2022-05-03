cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(learn_halide)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

find_package(Halide REQUIRED)

add_executable(learn_halide main.cpp)
target_link_libraries(learn_halide PRIVATE Halide::Halide)
