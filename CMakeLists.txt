cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(addr_rev LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

add_library(rbce_lib row_buffer_conflict_exp/rbce_n_conflict_exp.cu)
set_property(TARGET rbce_lib PROPERTY CUDA_ARCHITECTURES native)

add_executable(rbce ./row_buffer_conflict_exp/rbce_main.cu)
target_link_libraries(rbce PUBLIC rbce_lib)
set_property(TARGET rbce PROPERTY CUDA_ARCHITECTURES native)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info -O3 -Xcicc -O0 -Xptxas -O0")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info -O3")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")
