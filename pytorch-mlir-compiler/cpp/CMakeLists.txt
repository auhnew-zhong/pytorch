cmake_minimum_required(VERSION 3.18)
project(ai_compiler_mlir LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Python3
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Find Pybind11
find_package(pybind11 CONFIG)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found via find_package, trying to find via python")
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE pybind11_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(pybind11_DIR)
    find_package(pybind11 CONFIG REQUIRED)
  else()
    message(WARNING "pybind11 not found. Please ensure it is installed.")
  endif()
endif()

# Find MLIR
find_package(MLIR CONFIG REQUIRED)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)

add_library(_ai_compiler_mlir MODULE
  lib/Dialect.cpp
  lib/Passes.cpp
  lib/Pipeline.cpp
  lib/PyBind.cpp
)

set_target_properties(_ai_compiler_mlir PROPERTIES PREFIX "")

# MLIR is usually built without RTTI, so we must match that.
if(NOT LLVM_ENABLE_RTTI)
  target_compile_definitions(_ai_compiler_mlir PRIVATE PYBIND11_NO_RTTI)
  target_compile_options(_ai_compiler_mlir PRIVATE -fno-rtti)
endif()

target_include_directories(_ai_compiler_mlir PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${MLIR_INCLUDE_DIRS}
  ${LLVM_INCLUDE_DIRS}
)

target_link_libraries(_ai_compiler_mlir PRIVATE
  MLIRIR
  MLIRFuncDialect
  MLIRPass
  MLIRTransforms
  MLIRSupport
  pybind11::module
  Python3::Python
)
