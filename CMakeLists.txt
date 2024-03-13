cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(addr_rev LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# RBCE
add_library(rbce_lib row_buffer_conflict_exp/rbce_n_conflict_exp.cu)
set_property(TARGET rbce_lib PROPERTY CUDA_ARCHITECTURES native)

# add_executable(rbce ./row_buffer_conflict_exp/rbce_main.cu)
# target_link_libraries(rbce PUBLIC rbce_lib)
# set_property(TARGET rbce PROPERTY CUDA_ARCHITECTURES native)

# Get Bank Offset
add_executable(rbce_gbo ./row_buffer_conflict_exp/rbce_gbo_main.cu)
target_link_libraries(rbce_gbo PUBLIC rbce_lib)
set_property(TARGET rbce_gbo PROPERTY CUDA_ARCHITECTURES native)

# Get Timing values
add_executable(rbce_gen_time ./row_buffer_conflict_exp/rbce_gen_time_main.cu)
target_link_libraries(rbce_gen_time PUBLIC rbce_lib)
set_property(TARGET rbce_gen_time PROPERTY CUDA_ARCHITECTURES native)

# Hammer
## Simple hammer
add_executable(simple_hammer ./hammer/simple_hammer/main.cu)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info -O3 -Xcicc -O0 -Xptxas -O0")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info -O3")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --generate-line-info")
