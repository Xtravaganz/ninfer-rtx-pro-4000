cmake_minimum_required(VERSION 3.26)
project(ninfer_inkling2_stage3b_cuda LANGUAGES CXX CUDA)

# Standalone, opt-in CUDA target: does not touch the main NInfer CMake tree.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 120)
endif()
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

get_filename_component(NINFER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../.." ABSOLUTE)
find_package(CUDAToolkit REQUIRED)

add_executable(ninfer_inkling2_mxfp8_gemv_test
    "${NINFER_ROOT}/src/models/inkling2/mxfp8_gemv.cu"
    "${NINFER_ROOT}/tests/models/inkling2/test_mxfp8_gemv.cu")
target_include_directories(ninfer_inkling2_mxfp8_gemv_test PRIVATE "${NINFER_ROOT}/src")
target_link_libraries(ninfer_inkling2_mxfp8_gemv_test PRIVATE CUDA::cudart)
target_compile_options(ninfer_inkling2_mxfp8_gemv_test PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>)
set_target_properties(ninfer_inkling2_mxfp8_gemv_test PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    CUDA_SEPARABLE_COMPILATION OFF)

enable_testing()
add_test(NAME inkling2_mxfp8_native_cuda COMMAND ninfer_inkling2_mxfp8_gemv_test)
