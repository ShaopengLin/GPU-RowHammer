cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(addr_rev LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)


add_executable(timing timing.cu)
add_executable(t_warp timing_warp.cu)
set_property(TARGET timing PROPERTY CUDA_ARCHITECTURES native)
set_property(TARGET t_warp PROPERTY CUDA_ARCHITECTURES native)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info -O0 -Xcicc -O0 -Xptxas -O0")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info -O3")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")
