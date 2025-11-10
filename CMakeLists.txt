cmake_minimum_required(VERSION 3.16)
project(PiperBlas C CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenMP REQUIRED)
find_package(BLAS REQUIRED)
find_package(Eigen3 3.3 CONFIG REQUIRED)

file(GLOB_RECURSE LIB_SRC CONFIGURE_DEPENDS "src/*.c" "src/*.cpp")
add_library(piperblas STATIC ${LIB_SRC})
target_include_directories(piperblas PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
  "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(piperblas PUBLIC
  OpenMP::OpenMP_C
  OpenMP::OpenMP_CXX
  ${BLAS_LIBRARIES}
  Eigen3::Eigen)
target_compile_features(piperblas PUBLIC c_std_11 cxx_std_17)
target_compile_options(piperblas PRIVATE
  $<$<AND:$<C_COMPILER_ID:GNU,Clang>,$<CONFIG:Release>>:-O3 -ffast-math -march=native>
  $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<CONFIG:Release>>:-O3 -ffast-math -march=native>
  $<$<AND:$<C_COMPILER_ID:AppleClang>,$<CONFIG:Release>>:-O3 -ffast-math -mcpu=native>
  $<$<AND:$<CXX_COMPILER_ID:AppleClang>,$<CONFIG:Release>>:-O3 -ffast-math -mcpu=native>
  $<$<AND:$<C_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2>
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2>)

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_ok)
if(ipo_ok)
  set_property(TARGET piperblas PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

set(APP_MAIN "" CACHE STRING "Path to main source file")
if(APP_MAIN)
  add_executable(piper-blas ${APP_MAIN})
  target_link_libraries(piper-blas PRIVATE piperblas)
  target_compile_options(piper-blas PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<CONFIG:Release>>:-O3 -ffast-math -march=native>
    $<$<AND:$<CXX_COMPILER_ID:AppleClang>,$<CONFIG:Release>>:-O3 -ffast-math -mcpu=native>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2>)
  if(ipo_ok)
    set_property(TARGET piper-blas PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  endif()
endif()

file(GLOB TEST_SOURCES_CPP "tests/*.cpp")
foreach(test_src ${TEST_SOURCES_CPP})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_include_directories(${test_name} PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(${test_name} PRIVATE piperblas)
  target_compile_options(${test_name} PRIVATE
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang>,$<CONFIG:Release>>:-O3 -ffast-math -march=native>
    $<$<AND:$<CXX_COMPILER_ID:AppleClang>,$<CONFIG:Release>>:-O3 -ffast-math -mcpu=native>
    $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2>)
  if(ipo_ok)
    set_property(TARGET ${test_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
  endif()
endforeach()
