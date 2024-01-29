cmake_minimum_required(VERSION 3.27)
project(gddr CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_executable(gddr main.cu)

target_link_libraries(gddr cuda)

set_target_properties(gddr PROPERTIES CUDA_ARCHITECTURES "89")

set_target_properties(gddr PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
