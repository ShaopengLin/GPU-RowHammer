cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
# Initialize for C++ and CUDA languages.
project(addr_rev LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)


add_executable(op vm_map_id_op.cu)
set_property(TARGET op PROPERTY CUDA_ARCHITECTURES native)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O0 -Xcicc -O0 -Xptxas -O0")
