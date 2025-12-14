cmake_minimum_required(VERSION 3.20)
project(PiperDNN LANGUAGES CXX CUDA)

# =========================
# 基础配置
# =========================
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(PIPER_ENABLE_CUDA "Enable CUDA support" ON)
option(PIPER_BUILD_TESTS "Build tests" ON)

set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86")

# =========================
# 编译优化选项
# =========================
set(PIPER_OPT_FLAGS
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<CONFIG:Release>>:-O3 -ffast-math -march=native>
    $<$<AND:$<CXX_COMPILER_ID:AppleClang>,$<CONFIG:Release>>:-O3 -ffast-math -mcpu=native>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2>
)

# =========================
# 依赖
# =========================
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(CUDAToolkit REQUIRED)

# =========================
# MKL（可选、隔离）
# =========================
set(MKL_ROOT "/opt/intel/oneapi/mkl/latest" CACHE PATH "MKL root path")

if(EXISTS "${MKL_ROOT}")
    message(STATUS "MKL found at ${MKL_ROOT}")
    add_library(piper_mkl INTERFACE)
    target_include_directories(piper_mkl INTERFACE
        ${MKL_ROOT}/include
    )
    target_link_directories(piper_mkl INTERFACE
        ${MKL_ROOT}/lib/intel64
    )
    target_link_libraries(piper_mkl INTERFACE
        mkl_rt pthread m dl
    )
    set(PIPER_HAS_MKL ON)
else()
    message(STATUS "MKL not found, MKL-dependent tests will be skipped")
    set(PIPER_HAS_MKL OFF)
endif()

# =========================
# CPU 核心库
# =========================
file(GLOB CPU_SRC CONFIGURE_DEPENDS "src/*.cpp")

add_library(piper_cpu STATIC ${CPU_SRC})
target_include_directories(piper_cpu PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(piper_cpu PUBLIC
    Threads::Threads
    OpenMP::OpenMP_CXX
    BLAS::BLAS
)
target_compile_options(piper_cpu PRIVATE ${PIPER_OPT_FLAGS})

# =========================
# CUDA 核心库
# =========================
if(PIPER_ENABLE_CUDA)
    file(GLOB CUDA_SRC CONFIGURE_DEPENDS "src/cuda/*.cu" "src/cuda/*.cuh")

    add_library(piper_cuda STATIC ${CUDA_SRC})
    target_include_directories(piper_cuda PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/src
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CUDAToolkit_INCLUDE_DIRS}
    )
    target_link_libraries(piper_cuda PUBLIC
        piper_cpu
        CUDA::cudart
        CUDA::cublas
        CUDA::cusparse
    )
    target_compile_options(piper_cuda PRIVATE ${PIPER_OPT_FLAGS})
    set_target_properties(piper_cuda PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
endif()

# =========================
# 主程序（可选）
# =========================
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    add_executable(piper_main src/main.cpp)
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu")
    add_executable(piper_main src/main.cu)
endif()

if(TARGET piper_main)
    target_link_libraries(piper_main PRIVATE
        piper_cpu
        $<$<BOOL:${PIPER_ENABLE_CUDA}>:piper_cuda>
    )
    target_compile_options(piper_main PRIVATE ${PIPER_OPT_FLAGS})
endif()

# =========================
# Tests
# =========================
if(PIPER_BUILD_TESTS)
    file(GLOB TEST_CPP "tests/*.cpp")
    file(GLOB TEST_CU  "tests/*.cu")
    set(TEST_SOURCES ${TEST_CPP} ${TEST_CU})

    foreach(test_src ${TEST_SOURCES})
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})

        target_include_directories(${test_name} PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${CUDAToolkit_INCLUDE_DIRS}
        )

        target_link_libraries(${test_name} PRIVATE
            piper_cpu
            $<$<BOOL:${PIPER_ENABLE_CUDA}>:piper_cuda>
            CUDA::cudart
            OpenMP::OpenMP_CXX
        )

        target_link_libraries(${test_name} PRIVATE piper_mkl)

        target_compile_options(${test_name} PRIVATE ${PIPER_OPT_FLAGS})
    endforeach()
endif()
