cmake_minimum_required(VERSION 3.24)

project(forge_llm VERSION 0.1.0 LANGUAGES CXX)

option(FORGE_ENABLE_CUDA "Build the NVIDIA CUDA execution backend" ON)
option(FORGE_BUILD_PYTHON "Build the pybind11 Python extension" ON)
option(FORGE_BUILD_TESTS "Build tests" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(forge_host
    src/model_file.cpp
    src/qwen_validation.cpp
    src/sha256.cpp
    src/kv_cache.cpp
    src/scheduler.cpp)
target_include_directories(forge_host PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_options(forge_host PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:-Wall;-Wextra;-Wpedantic;-Wconversion>)

add_executable(forge-inspect-model tools/inspect_model.cpp)
target_link_libraries(forge-inspect-model PRIVATE forge_host)

include(CheckLanguage)
if(FORGE_ENABLE_CUDA)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit 12.0 REQUIRED)
    add_library(forge_cuda
        cuda/cuda_arena.cu
        cuda/execution_context.cu
        cuda/kernels.cu
        cuda/model_weights.cu
        cuda/qwen_model.cu
        cuda/engine.cu)
    target_include_directories(forge_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(forge_cuda PUBLIC forge_host CUDA::cudart CUDA::cublasLt CUDA::nvToolsExt)
    target_compile_definitions(forge_cuda PUBLIC FORGE_HAS_CUDA=1)
    set_target_properties(forge_cuda PROPERTIES
        CUDA_ARCHITECTURES "86"
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        POSITION_INDEPENDENT_CODE ON)
  else()
    message(STATUS "CUDA compiler not found; building portable host components only")
    set(FORGE_ENABLE_CUDA OFF)
  endif()
endif()

if(FORGE_BUILD_PYTHON AND FORGE_ENABLE_CUDA)
  find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_forge python/bindings.cpp)
  target_link_libraries(_forge PRIVATE forge_cuda)
  install(TARGETS _forge LIBRARY DESTINATION forge_llm)
endif()

if(FORGE_BUILD_TESTS)
  enable_testing()
  add_executable(forge_host_tests tests/host_tests.cpp)
  target_link_libraries(forge_host_tests PRIVATE forge_host)
  add_test(NAME forge_host_tests COMMAND forge_host_tests)
  if(FORGE_ENABLE_CUDA)
    add_executable(forge_cuda_tests tests/cuda_tests.cu)
    target_link_libraries(forge_cuda_tests PRIVATE forge_cuda)
    set_target_properties(forge_cuda_tests PROPERTIES CUDA_ARCHITECTURES "86" CUDA_STANDARD 20)
    add_test(NAME forge_cuda_tests COMMAND forge_cuda_tests)
  endif()
endif()
