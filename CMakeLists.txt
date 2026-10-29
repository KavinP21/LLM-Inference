cmake_minimum_required(VERSION 3.24)

project(instinct VERSION 0.1.0 LANGUAGES CXX)

option(INSTINCT_ENABLE_CUDA "Build the NVIDIA CUDA execution backend" ON)
option(INSTINCT_BUILD_PYTHON "Build the pybind11 Python extension" ON)
option(INSTINCT_BUILD_TESTS "Build tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(instinct_host
    src/model_file.cpp
    src/model_validation.cpp
    src/qwen_validation.cpp
    src/sha256.cpp
    src/kv_cache.cpp
    src/scheduler.cpp)
target_include_directories(instinct_host PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
# Linked into the _instinct Python module, so it has to be PIC on Linux too.
set_target_properties(instinct_host PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_compile_options(instinct_host PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Wconversion>)

add_executable(instinct-inspect-model tools/inspect_model.cpp)
target_link_libraries(instinct-inspect-model PRIVATE instinct_host)

include(CheckLanguage)
if(INSTINCT_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit 12.0 REQUIRED)
    add_library(instinct_cuda
        cuda/cuda_arena.cu
        cuda/execution_context.cu
        cuda/kernels.cu
        cuda/model_weights.cu
        cuda/qwen_model.cu
        cuda/engine.cu)
    target_include_directories(instinct_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(instinct_cuda PUBLIC instinct_host CUDA::cudart CUDA::cublasLt CUDA::nvToolsExt)
    target_compile_definitions(instinct_cuda PUBLIC INSTINCT_HAS_CUDA=1)
    set_target_properties(instinct_cuda PROPERTIES
        CUDA_ARCHITECTURES "86"
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON)
  else()
    message(STATUS "CUDA compiler not found; building portable host components only")
    set(INSTINCT_ENABLE_CUDA OFF)
  endif()
endif()

if(INSTINCT_BUILD_PYTHON AND INSTINCT_ENABLE_CUDA)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_instinct python/bindings.cpp)
  target_link_libraries(_instinct PRIVATE instinct_cuda)
  install(TARGETS _instinct LIBRARY DESTINATION instinct)
endif()

if(INSTINCT_BUILD_TESTS)
  enable_testing()
  add_executable(instinct_host_tests tests/host_tests.cpp)
  target_link_libraries(instinct_host_tests PRIVATE instinct_host)
  add_test(NAME instinct_host_tests COMMAND instinct_host_tests)
  if(INSTINCT_ENABLE_CUDA)
    add_executable(instinct_cuda_tests tests/cuda_tests.cu)
    target_link_libraries(instinct_cuda_tests PRIVATE instinct_cuda)
    set_target_properties(instinct_cuda_tests PROPERTIES CUDA_ARCHITECTURES "86" CUDA_STANDARD 20)
    add_test(NAME instinct_cuda_tests COMMAND instinct_cuda_tests)
  endif()
endif()
