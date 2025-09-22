cmake_minimum_required(VERSION 3.16)
project(PiperBlas C CXX)

find_package(BLAS REQUIRED)

set(OPENBLAS_ROOT "/opt/homebrew/opt/openblas")

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS "src/*.c" "src/*.cpp")

add_executable(piper-blas ${SRC_FILES})
target_include_directories(piper-blas PRIVATE "${OPENBLAS_ROOT}/include")
target_link_directories(piper-blas PRIVATE "${OPENBLAS_ROOT}/lib")
target_link_libraries(piper-blas PRIVATE ${BLAS_LIBRARIES})  

add_executable(gemm_test tests/gemm_test.cpp)
target_include_directories(gemm_test PRIVATE "${OPENBLAS_ROOT}/include")
target_link_directories(gemm_test PRIVATE "${OPENBLAS_ROOT}/lib")
target_link_libraries(gemm_test PRIVATE ${BLAS_LIBRARIES})