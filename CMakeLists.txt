cmake_minimum_required(VERSION 3.20)
project(PiperDNN LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(PIPER_ENABLE_CUDA ON)
option(PIPER_BUILD_TESTS ON)

set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(MKL_ROOT "/opt/intel/oneapi/mkl/latest")
set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
set(MKL_LIBRARY_DIR "${MKL_ROOT}/lib/intel64")
set(MKL_LIBRARIES mkl_rt pthread m dl)

set(PIPER_OPT_FLAGS
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<CONFIG:Release>>:-O3 -ffast-math -march=native>
    $<$<AND:$<CXX_COMPILER_ID:AppleClang>,$<CONFIG:Release>>:-O3 -ffast-math -mcpu=native>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2>
    $<$<AND:$<C_COMPILER_ID:GNU,Clang>,$<CONFIG:Release>>:-O3 -ffast-math -march=native>
    $<$<AND:$<C_COMPILER_ID:AppleClang>,$<CONFIG:Release>>:-O3 -ffast-math -mcpu=native>
    $<$<AND:$<C_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2>
)

file(GLOB CPU_SRC CONFIGURE_DEPENDS "src/*.cpp")
add_library(piperDNN STATIC ${CPU_SRC})
target_include_directories(piperDNN PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(piperDNN PUBLIC
    Threads::Threads
    OpenMP::OpenMP_CXX
    BLAS::BLAS)
target_compile_options(piperDNN PRIVATE ${PIPER_OPT_FLAGS})

file(GLOB CUDA_SRC CONFIGURE_DEPENDS "src/cuda/*.cu" "src/cuda/*.cuh")
add_library(pipercuda STATIC ${CUDA_SRC})
target_include_directories(pipercuda PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(pipercuda PUBLIC
    piperDNN
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse)
target_compile_options(pipercuda PRIVATE ${PIPER_OPT_FLAGS})
set_target_properties(pipercuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

set(MAIN_SOURCE "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu")
    set(MAIN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    set(MAIN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
endif()

add_executable(piper-DNN ${MAIN_SOURCE})
target_link_libraries(piper-DNN PRIVATE
    piperDNN
    pipercuda
    CUDA::cudart)
target_compile_options(piper-DNN PRIVATE ${PIPER_OPT_FLAGS})
set_target_properties(piper-DNN PROPERTIES
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")

file(GLOB TEST_SOURCES_CPP "tests/*.cpp")
file(GLOB TEST_SOURCES_CU  "tests/*.cu")
set(TEST_SOURCES ${TEST_SOURCES_CPP} ${TEST_SOURCES_CU})

foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name} PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        ${CUDAToolkit_INCLUDE_DIRS})
    target_link_libraries(${test_name} PRIVATE 
        piperDNN
        pipercuda
        CUDA::cudart
        CUDA::cublas
        OpenMP::OpenMP_CXX)
    target_compile_options(${test_name} PRIVATE ${PIPER_OPT_FLAGS})

    if(test_name STREQUAL "test_spmv")
        target_include_directories(${test_name} PRIVATE "${MKL_INCLUDE_DIR}")
        target_link_directories(${test_name} PRIVATE "${MKL_LIBRARY_DIR}")
        target_link_libraries(${test_name} PRIVATE ${MKL_LIBRARIES})
    endif()

    if(test_name STREQUAL "test_gemm")
        target_link_libraries(${test_name} PRIVATE BLAS::BLAS)
    endif()
endforeach()
