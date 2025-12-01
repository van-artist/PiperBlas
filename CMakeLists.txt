cmake_minimum_required(VERSION 3.16)
project(PiperDNN C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

set(MKL_ROOT "/opt/intel/oneapi/mkl/latest" CACHE PATH "Path to Intel oneMKL root")
set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
set(MKL_LIBRARY_DIR "${MKL_ROOT}/lib/intel64")
set(MKL_LIBRARIES mkl_rt pthread m dl)
find_package(BLAS REQUIRED)


file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS "src/*.c" "src/*.cpp")

add_library(piperDNN STATIC ${SRC_FILES})
target_include_directories(piperDNN PUBLIC
    "${OPENBLAS_ROOT}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${MKL_INCLUDE_DIR}
)
target_link_directories(piperDNN PUBLIC "${OPENBLAS_ROOT}/lib")
target_link_libraries(piperDNN PUBLIC ${BLAS_LIBRARIES})

add_executable(piper-blas ${SRC_FILES})
target_link_libraries(piper-blas PRIVATE piperDNN)

file(GLOB TEST_SOURCES "tests/*.cpp")
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name} PRIVATE
        "${OPENBLAS_ROOT}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
        ${MKL_INCLUDE_DIR}
    )
    target_link_directories(${test_name} PRIVATE "${OPENBLAS_ROOT}/lib")
    target_link_libraries(${test_name} PRIVATE piperDNN)
endforeach()