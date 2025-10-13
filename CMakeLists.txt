cmake_minimum_required(VERSION 3.16)
project(PiperBlas C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(OPENBLAS_ROOT "/opt/homebrew/opt/openblas")
set(Eigen3_DIR "/opt/homebrew/lib/cmake/eigen3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

find_package(BLAS REQUIRED)
find_package(Eigen3 3.3 CONFIG REQUIRED)

file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS "src/*.c" "src/*.cpp")

add_library(piperblas STATIC ${SRC_FILES})
target_include_directories(piperblas PUBLIC
    "${OPENBLAS_ROOT}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
target_link_directories(piperblas PUBLIC "${OPENBLAS_ROOT}/lib")
target_link_libraries(piperblas PUBLIC ${BLAS_LIBRARIES})

add_executable(piper-blas ${SRC_FILES})
target_link_libraries(piper-blas PRIVATE piperblas)

file(GLOB TEST_SOURCES "tests/*.cpp")
foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_include_directories(${test_name} PRIVATE
        "${OPENBLAS_ROOT}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/include"
        "${CMAKE_CURRENT_SOURCE_DIR}/src"
    )
    target_link_directories(${test_name} PRIVATE "${OPENBLAS_ROOT}/lib")
    target_link_libraries(${test_name} PRIVATE piperblas Eigen3::Eigen)
endforeach()