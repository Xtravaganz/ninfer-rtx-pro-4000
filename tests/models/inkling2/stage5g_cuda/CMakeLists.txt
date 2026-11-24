cmake_minimum_required(VERSION 3.26)
project(ninfer_inkling2_stage5g LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
find_package(CUDAToolkit REQUIRED)
add_executable(ninfer_inkling2_real_moe_stage5g
  ${NINFER_ROOT}/src/models/inkling2/moe_top8_stage5e.cu
  ${NINFER_ROOT}/tests/models/inkling2/stage5g_cuda/test_real_moe_stage5g.cu)
target_include_directories(ninfer_inkling2_real_moe_stage5g PRIVATE ${NINFER_ROOT}/src)
target_link_libraries(ninfer_inkling2_real_moe_stage5g PRIVATE CUDA::cudart)
set_target_properties(ninfer_inkling2_real_moe_stage5g PROPERTIES CUDA_SEPARABLE_COMPILATION OFF)
target_compile_options(ninfer_inkling2_real_moe_stage5g PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--ftz=false>)
