cmake_minimum_required(VERSION 3.20)
project(PiperDNN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(PIPER_ENABLE_CUDA "Build CUDA kernels" ON)
option(PIPER_BUILD_TESTS "Build test binaries" ON)

set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86" CACHE STRING "CUDA GPU architectures")
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

find_package(Threads REQUIRED)
find_package(BLAS REQUIRED)

set(MKL_ROOT "/opt/intel/oneapi/mkl/latest" CACHE PATH "Path to Intel oneMKL root")
set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
set(MKL_LIBRARY_DIR "${MKL_ROOT}/lib/intel64")
set(MKL_LIBRARIES mkl_rt pthread m dl)

if(PIPER_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)
endif()

file(GLOB CPU_SRC CONFIGURE_DEPENDS "src/*.cpp")

add_library(piperDNN STATIC ${CPU_SRC})
target_compile_features(piperDNN PUBLIC cxx_std_17)
target_include_directories(piperDNN PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_libraries(piperDNN PUBLIC Threads::Threads)

if(PIPER_ENABLE_CUDA)
    file(GLOB CUDA_SRC CONFIGURE_DEPENDS
        "src/cuda/*.cu"
        "src/cuda/*.cuh"
    )

    if(CUDA_SRC)
        add_library(pipercuda STATIC ${CUDA_SRC})
        target_compile_features(pipercuda PUBLIC cxx_std_17)
        target_include_directories(pipercuda PUBLIC
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            ${CUDAToolkit_INCLUDE_DIRS}
        )
        target_link_libraries(pipercuda PUBLIC
            piperDNN
            CUDA::cudart
            CUDA::cublas
            CUDA::cusparse
        )
        set_target_properties(pipercuda PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        )
    endif()
endif()

set(MAIN_SOURCE "")
if(PIPER_ENABLE_CUDA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu")
    set(MAIN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu")
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
    set(MAIN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
endif()

if(MAIN_SOURCE)
    add_executable(piper-DNN ${MAIN_SOURCE})
    target_link_libraries(piper-DNN PRIVATE piperDNN)
    if(PIPER_ENABLE_CUDA AND TARGET pipercuda)
        target_link_libraries(piper-DNN PRIVATE pipercuda CUDA::cudart)
        set_target_properties(piper-DNN PROPERTIES
            CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
        )
    endif()
endif()

if(PIPER_BUILD_TESTS)
    file(GLOB TEST_SOURCES "tests/*.cpp")
    if(PIPER_ENABLE_CUDA)
        file(GLOB TEST_SOURCES_CUDA "tests/*.cu")
        list(APPEND TEST_SOURCES ${TEST_SOURCES_CUDA})
    endif()

    foreach(test_src IN LISTS TEST_SOURCES)
        get_filename_component(test_name ${test_src} NAME_WE)
        add_executable(${test_name} ${test_src})

        target_include_directories(${test_name} PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/src"
            $<$<BOOL:${PIPER_ENABLE_CUDA}>:${CUDAToolkit_INCLUDE_DIRS}>
        )

        if(test_name STREQUAL "test_gemm")
            target_link_libraries(${test_name} PRIVATE BLAS::BLAS)
        endif()

        if(test_name STREQUAL "test_spmv")
            target_include_directories(${test_name} PRIVATE "${MKL_INCLUDE_DIR}")
            target_link_directories(${test_name} PRIVATE "${MKL_LIBRARY_DIR}")
            target_link_libraries(${test_name} PRIVATE ${MKL_LIBRARIES})
        endif()

        target_link_libraries(${test_name} PRIVATE piperDNN)

        if(PIPER_ENABLE_CUDA AND TARGET pipercuda)
            target_link_libraries(${test_name} PRIVATE
                pipercuda
                CUDA::cudart
                CUDA::cublas
            )
        endif()
    endforeach()
endif()
