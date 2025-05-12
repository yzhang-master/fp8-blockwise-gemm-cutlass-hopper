cmake_minimum_required(VERSION 3.24)
project(fp8_blockwise_sm90 LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90a)
set(CUTLASS_DIR "" CACHE PATH "Path to a CUTLASS 4.x checkout")
if(NOT EXISTS "${CUTLASS_DIR}/include/cutlass/cutlass.h")
  message(FATAL_ERROR "Set CUTLASS_DIR to a CUTLASS checkout containing include/cutlass")
endif()

find_package(CUDAToolkit REQUIRED)
add_executable(fp8_blockwise_sm90 fp8_blockwise_sm90.cu)
target_include_directories(fp8_blockwise_sm90 PRIVATE
  "${CUTLASS_DIR}/include"
  "${CUTLASS_DIR}/examples/common"
  "${CUTLASS_DIR}/tools/util/include"
)
target_link_libraries(fp8_blockwise_sm90 PRIVATE CUDA::cudart CUDA::cublas)
target_compile_options(fp8_blockwise_sm90 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr;-O3>
)
