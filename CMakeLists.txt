cmake_minimum_required(VERSION 3.20)

project(PiperRT LANGUAGES CXX)

if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

option(PIPER_BUILD_TESTS       "Build tests" ON)
option(PIPER_ENABLE_MKL        "Enable MKL detection (for MKL tests/paths)" ON)
option(PIPER_ENABLE_EIGEN      "Enable Eigen support (for Eigen-based tests)" ON)

set(PIPER_ENABLE_CPU_BLAS ON)
set(PIPER_ENABLE_CUDA ON)

set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86" CACHE STRING "CUDA architectures")

set(PIPER_OPT_FLAGS
  $<$<AND:$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>,$<CONFIG:Release>>:-O3 -ffast-math -march=native>
  $<$<AND:$<CXX_COMPILER_ID:AppleClang>,$<CONFIG:Release>>:-O3 -ffast-math -mcpu=native>
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2>
)

find_package(Threads REQUIRED)

set(PIPER_HAVE_EIGEN OFF)
if(PIPER_ENABLE_EIGEN)
  find_package(Eigen3 QUIET)
  if(Eigen3_FOUND)
    set(PIPER_HAVE_EIGEN ON)
  else()
    message(STATUS "Eigen3 not found; Eigen-dependent components will be skipped")
  endif()
else()
  message(STATUS "Eigen support disabled via PIPER_ENABLE_EIGEN=OFF")
endif()

add_library(piper_rt_build INTERFACE)
target_link_libraries(piper_rt_build INTERFACE Threads::Threads)
target_compile_options(piper_rt_build INTERFACE ${PIPER_OPT_FLAGS})
target_compile_features(piper_rt_build INTERFACE cxx_std_17)

option(PIPER_ENABLE_MPI "Enable MPI support" ON)
set(PIPER_HAVE_MPI OFF)
if(PIPER_ENABLE_MPI)
  find_package(MPI QUIET)
  if(MPI_FOUND)
    set(PIPER_HAVE_MPI ON)
    target_link_libraries(piper_rt_build INTERFACE MPI::MPI_CXX)
  else()
    message(WARNING "MPI requested but not found; MPI support disabled")
  endif()
endif()

set(PIPER_HAVE_MKL OFF)

set(MKL_ROOT "" CACHE PATH "Intel oneAPI MKL root")
if(PIPER_ENABLE_MKL)
  if(NOT MKL_ROOT AND DEFINED ENV{MKLROOT})
    set(MKL_ROOT "$ENV{MKLROOT}")
  endif()

  find_path(MKL_INCLUDE_DIR NAMES mkl.h HINTS "${MKL_ROOT}" PATH_SUFFIXES include)
  find_library(MKL_RT_LIB NAMES mkl_rt HINTS "${MKL_ROOT}" PATH_SUFFIXES lib lib/intel64 intel64 lib/x64)

  if(MKL_INCLUDE_DIR AND MKL_RT_LIB)
    set(PIPER_HAVE_MKL ON)
    set(MKL_LIBRARIES "${MKL_RT_LIB}")
    if(UNIX AND NOT APPLE)
      list(APPEND MKL_LIBRARIES pthread m dl)
    endif()
    add_library(Piper::MKL INTERFACE IMPORTED)
    target_include_directories(Piper::MKL INTERFACE "${MKL_INCLUDE_DIR}")
    target_link_libraries(Piper::MKL INTERFACE ${MKL_LIBRARIES})
  endif()
endif()

set(PIPER_HAVE_BLAS OFF)

if(PIPER_ENABLE_CPU_BLAS)
  find_package(BLAS QUIET)

  if(BLAS_FOUND)
    find_path(PIPER_CBLAS_INCLUDE_DIR
      NAMES cblas.h
      HINTS "${PIPER_OPENBLAS_ROOT}" "$ENV{OpenBLAS_HOME}" "$ENV{OPENBLAS_ROOT}" "$ENV{CONDA_PREFIX}" "/usr"
      PATH_SUFFIXES include include/openblas include/x86_64-linux-gnu include/x86_64-linux-gnu/openblas-pthread
    )

    add_library(Piper::CBLAS INTERFACE IMPORTED)
    if(PIPER_CBLAS_INCLUDE_DIR)
      target_include_directories(Piper::CBLAS INTERFACE "${PIPER_CBLAS_INCLUDE_DIR}")
    endif()
    target_link_libraries(Piper::CBLAS INTERFACE BLAS::BLAS)
    set(PIPER_HAVE_BLAS ON)
  else()
    message(STATUS "BLAS not found; BLAS-dependent components will be skipped")
  endif()
else()
  message(STATUS "BLAS support disabled via PIPER_ENABLE_CPU_BLAS=OFF")
endif()

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDAToolkit REQUIRED)
set(PIPER_HAVE_CUDA ON)

add_library(piper_rt_core STATIC
  src/core/common.cpp
  src/core/pi_config.cpp
  src/core/pi_csr.cpp
  src/core/test_utils.cpp
  src/core/pi_cluster.cpp
)

add_library(piper-rt::core ALIAS piper_rt_core)

target_include_directories(piper_rt_core
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(piper_rt_core PUBLIC piper_rt_build)
if(PIPER_HAVE_CUDA)
  target_link_libraries(piper_rt_core PUBLIC CUDA::cudart)
endif()


target_compile_definitions(piper_rt_core PUBLIC
  PIPER_HAVE_BLAS=$<BOOL:${PIPER_HAVE_BLAS}>
  PIPER_HAVE_MKL=$<BOOL:${PIPER_HAVE_MKL}>
  PIPER_HAVE_MPI=$<BOOL:${PIPER_HAVE_MPI}>
  PIPER_HAVE_CUDA=$<BOOL:${PIPER_HAVE_CUDA}>
)

set(PIPER_BLAS_SRCS
  src/blas/pi_gemm.cpp
  src/blas/pi_spmv.cpp
)

if(PIPER_HAVE_MPI)
  list(APPEND PIPER_BLAS_SRCS src/blas/pi_summa.cpp)
else()
  message(STATUS "Skipping MPI-dependent source: src/blas/pi_summa.cpp")
endif()

add_library(piper_rt_blas STATIC ${PIPER_BLAS_SRCS})

add_library(piper-rt::blas ALIAS piper_rt_blas)

target_link_libraries(piper_rt_blas PUBLIC piper-rt::core)
if(PIPER_HAVE_BLAS)
  target_link_libraries(piper_rt_blas PUBLIC Piper::CBLAS)
endif()

file(GLOB_RECURSE DSP_SRC CONFIGURE_DEPENDS
  src/dsp/*.cpp
  src/dsp/*.cc
  src/dsp/*.cxx
)

if(DSP_SRC)
  add_library(piper_rt_dsp STATIC ${DSP_SRC})
  add_library(piper-rt::dsp ALIAS piper_rt_dsp)
  target_link_libraries(piper_rt_dsp PUBLIC piper-rt::core)
endif()


add_library(piper_rt_cuda STATIC
  src/cuda/axpy_kernels.cu
  src/cuda/gemm_kernel.cu
  src/cuda/spmv_kernels.cu
  src/cuda/cuda_common.cu
)

add_library(piper-rt::cuda ALIAS piper_rt_cuda)

target_include_directories(piper_rt_cuda
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(piper_rt_cuda PUBLIC
  piper-rt::core
  CUDA::cudart
  CUDA::cublas
  CUDA::cusparse
)

set_target_properties(piper_rt_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)

if(PIPER_BUILD_TESTS)
  file(GLOB TEST_SOURCES_CPP CONFIGURE_DEPENDS tests/*.cpp)
  file(GLOB TEST_SOURCES_CU  CONFIGURE_DEPENDS tests/*.cu)

  set(TEST_SPMV_CPU "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_spmv_cpu.cpp")
  set(TEST_CPU_GEMM "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_cpu_gemm.cpp")
  list(REMOVE_ITEM TEST_SOURCES_CPP "${TEST_SPMV_CPU}")
  list(REMOVE_ITEM TEST_SOURCES_CPP "${TEST_CPU_GEMM}")

  foreach(test_src ${TEST_SOURCES_CPP})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE piper-rt::blas)
    if(TARGET piper_rt_cuda)
      target_link_libraries(${test_name} PRIVATE piper-rt::cuda)
    endif()
  endforeach()

  if(PIPER_HAVE_EIGEN)
    add_executable(test_spmv_cpu "${TEST_SPMV_CPU}")
    target_link_libraries(test_spmv_cpu PRIVATE piper-rt::blas Eigen3::Eigen)
    if(TARGET piper_rt_cuda)
      target_link_libraries(test_spmv_cpu PRIVATE piper-rt::cuda)
    endif()
  else()
    message(STATUS "Skipping Eigen-dependent test: test_spmv_cpu")
  endif()

  if(PIPER_HAVE_BLAS)
    add_executable(test_cpu_gemm "${TEST_CPU_GEMM}")
    target_link_libraries(test_cpu_gemm PRIVATE piper-rt::blas Piper::CBLAS)
  else()
    message(STATUS "Skipping BLAS-dependent test: test_cpu_gemm")
  endif()

  if(PIPER_HAVE_CUDA)
    foreach(test_src ${TEST_SOURCES_CU})
      get_filename_component(test_name ${test_src} NAME_WE)
      add_executable(${test_name} ${test_src})
      target_link_libraries(${test_name} PRIVATE piper-rt::blas piper-rt::cuda)
      set_target_properties(${test_name} PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      )
    endforeach()
  endif()
endif()

file(GLOB BIN_SOURCES_CPP CONFIGURE_DEPENDS bin/*.cpp)
foreach(bin_src ${BIN_SOURCES_CPP})
  get_filename_component(bin_name ${bin_src} NAME_WE)
  add_executable(${bin_name} ${bin_src})
  target_link_libraries(${bin_name} PRIVATE piper-rt::blas)
  if(TARGET piper_rt_cuda)
    target_link_libraries(${bin_name} PRIVATE piper-rt::cuda)
  endif()
endforeach()

if(PIPER_HAVE_CUDA)
  file(GLOB BIN_SOURCES_CU CONFIGURE_DEPENDS bin/*.cu)
  foreach(bin_src ${BIN_SOURCES_CU})
    get_filename_component(bin_name ${bin_src} NAME_WE)
    add_executable(${bin_name} ${bin_src})
    target_link_libraries(${bin_name} PRIVATE piper-rt::blas piper-rt::cuda)
    set_target_properties(${bin_name} PROPERTIES
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
  endforeach()
endif()


message(STATUS "==== PiperRT Config Summary ====")
message(STATUS "PIPER_BUILD_TESTS      = ${PIPER_BUILD_TESTS}")
message(STATUS "PIPER_ENABLE_CUDA      = ${PIPER_ENABLE_CUDA}")
message(STATUS "PIPER_HAVE_CUDA        = ${PIPER_HAVE_CUDA}")
message(STATUS "PIPER_ENABLE_CPU_BLAS  = ${PIPER_ENABLE_CPU_BLAS}")
message(STATUS "PIPER_HAVE_BLAS        = ${PIPER_HAVE_BLAS}")
message(STATUS "PIPER_ENABLE_MKL       = ${PIPER_ENABLE_MKL}")
message(STATUS "PIPER_HAVE_MKL         = ${PIPER_HAVE_MKL}")
message(STATUS "PIPER_ENABLE_MPI       = ${PIPER_ENABLE_MPI}")
message(STATUS "PIPER_HAVE_MPI         = ${PIPER_HAVE_MPI}")
message(STATUS "PIPER_ENABLE_EIGEN     = ${PIPER_ENABLE_EIGEN}")
message(STATUS "PIPER_HAVE_EIGEN       = ${PIPER_HAVE_EIGEN}")
message(STATUS "OpenBLAS lib           = ${PIPER_OPENBLAS_LIB}")
message(STATUS "MKL_ROOT               = ${MKL_ROOT}")
message(STATUS "================================")
