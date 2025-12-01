cmake_minimum_required(VERSION 3.20)
project(PiperDNN LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

option(PIPER_ENABLE_CUDA "Build CUDA kernels" ON)

find_package(BLAS REQUIRED)
find_package(Threads REQUIRED)

set(MKL_ROOT "/opt/intel/oneapi/mkl/latest" CACHE PATH "Path to Intel oneMKL root")
set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
set(MKL_LIBRARY_DIR "${MKL_ROOT}/lib/intel64")
set(MKL_LIBRARIES mkl_rt pthread m dl)

file(GLOB_RECURSE CPU_SRC CONFIGURE_DEPENDS "src/*.cpp")

add_library(piperDNN STATIC ${CPU_SRC})
target_include_directories(piperDNN PUBLIC
    "${OPENBLAS_ROOT}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${MKL_INCLUDE_DIR}"
)
target_link_directories(piperDNN PUBLIC
    "${OPENBLAS_ROOT}/lib"
    "${MKL_LIBRARY_DIR}"
)
target_link_libraries(piperDNN PUBLIC
    ${BLAS_LIBRARIES}
    ${MKL_LIBRARIES}
    Threads::Threads
)

if(PIPER_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)

    file(GLOB_RECURSE CUDA_SRC CONFIGURE_DEPENDS
        "src/cuda/*.cu"
        "src/cuda/*.cuh"
    )

    if(CUDA_SRC)
        add_library(pipercuda STATIC ${CUDA_SRC})
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
            CUDA_ARCHITECTURES "70;75;80;86"
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
    add_executable(piper-DNN ${CPU_SRC} ${MAIN_SOURCE})
    target_link_libraries(piper-DNN PRIVATE piperDNN)
    if(PIPER_ENABLE_CUDA AND TARGET pipercuda)
        target_link_libraries(piper-DNN PRIVATE pipercuda CUDA::cudart)
    endif()
endif()

file(GLOB TEST_SOURCES "tests/*.cpp")
if(PIPER_ENABLE_CUDA)
    file(GLOB TEST_SOURCES_CUDA "tests/*.cu")
    list(APPEND TEST_SOURCES ${TEST_SOURCES_CUDA})
endif()
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})

    target_include_directories(${test_name} PRIVATE
        "${OPENBLAS_ROOT}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        "${MKL_INCLUDE_DIR}"
        $<$<BOOL:${PIPER_ENABLE_CUDA}>:${CUDAToolkit_INCLUDE_DIRS}>
    )

    target_link_directories(${test_name} PRIVATE
        "${OPENBLAS_ROOT}/lib"
        "${MKL_LIBRARY_DIR}"
    )

    target_link_libraries(${test_name} PRIVATE
        piperDNN
        ${MKL_LIBRARIES}
        $<$<TARGET_EXISTS:pipercuda>:pipercuda>
        $<$<TARGET_EXISTS:pipercuda>:CUDA::cudart>
    )
endforeach()
