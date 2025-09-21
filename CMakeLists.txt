cmake_minimum_required(VERSION 3.17)

project(mytorch-cuda-backend LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)

add_custom_target(
    run_python_script
    ALL
    COMMAND cd ${CMAKE_SOURCE_DIR} && "${PYTHON}" -m mytorch.backends.cuda.generate_cuda_sources
    ARGS    
    BYPRODUCTS
    COMMENT "Running Python script to generate cuda sources..."
)

file(GLOB CUDA_SOURCES "generated/*.cu")
add_library("mytorch-cuda-backend" SHARED "${CUDA_SOURCES}")
target_include_directories("mytorch-cuda-backend" PRIVATE "mytorch/native/cuda")
add_dependencies("mytorch-cuda-backend" run_python_script)
