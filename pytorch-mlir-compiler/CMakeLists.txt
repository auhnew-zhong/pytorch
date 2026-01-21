cmake_minimum_required(VERSION 3.15)
project(pytorch_mlir_compiler LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(MLIR REQUIRED CONFIG)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_subdirectory(cpp)

