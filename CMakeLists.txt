cmake_minimum_required(VERSION 3.16)
project(PiperBlas C CXX)

find_package(BLAS REQUIRED)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    "${CMAKE_SOURCE_DIR}/src/*.c"
    "${CMAKE_SOURCE_DIR}/src/*.cpp"
)

add_executable(piper-blas ${SRC_FILES})

target_include_directories(piper-blas PRIVATE ${CMAKE_SOURCE_DIR}/include)

target_link_libraries(piper-blas PRIVATE ${BLAS_LIBRARIES})