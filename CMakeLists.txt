cmake_minimum_required(VERSION 3.20)

# 先只声明 CXX，CUDA 由 PIPER_ENABLE_CUDA 控制按需启用
project(PiperDNN LANGUAGES CXX)

# =========================
# 基础配置
# =========================
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# =========================
# 选项开关
# =========================
option(PIPER_ENABLE_CUDA "Build CUDA backend (cuBLAS/cuSPARSE)" ON)
option(PIPER_BUILD_TESTS "Build tests" ON)

# CPU BLAS（系统 BLAS，如 OpenBLAS/ATLAS/MKL/Reference BLAS）
option(PIPER_ENABLE_CPU_BLAS "Use CPU BLAS if available" OFF)
option(PIPER_REQUIRE_CPU_BLAS "Fail if CPU BLAS requested but missing" OFF)

# MKL（如果你需要的话保留）
option(PIPER_ENABLE_MKL "Enable MKL detection (for specific tests/paths)" OFF)

# GPU 架构（按需改）
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86" CACHE STRING "CUDA architectures")

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# =========================
# 编译优化选项
# =========================
set(PIPER_OPT_FLAGS
  $<$<AND:$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>,$<CONFIG:Release>>:-O3 -ffast-math -march=native>
  $<$<AND:$<CXX_COMPILER_ID:AppleClang>,$<CONFIG:Release>>:-O3 -ffast-math -mcpu=native>
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2>
)

# =========================
# 依赖：通用（CPU）
# =========================
find_package(Threads REQUIRED)
find_package(OpenMP REQUIRED)

# =========================
# CPU BLAS：按需查找（默认不找，避免 GPU 平台无 BLAS 报错）
# =========================
set(PIPER_HAVE_BLAS OFF)
if(PIPER_ENABLE_CPU_BLAS)
  # 有些集群环境会“全局强制 BLAS 必须找到”，不进 FindBLAS 是最稳的
  find_package(BLAS QUIET)
  if(BLAS_FOUND)
    set(PIPER_HAVE_BLAS ON)
  elseif(PIPER_REQUIRE_CPU_BLAS)
    message(FATAL_ERROR "CPU BLAS requested but not found. Install BLAS or set -DPIPER_ENABLE_CPU_BLAS=OFF.")
  endif()
endif()

# =========================
# MKL：按需启用
# =========================
set(PIPER_HAVE_MKL OFF)
if(PIPER_ENABLE_MKL)
  if(NOT DEFINED MKL_ROOT AND DEFINED ENV{MKLROOT})
    set(MKL_ROOT $ENV{MKLROOT})
  endif()

  if(MKL_ROOT)
    set(MKL_INCLUDE_DIR "${MKL_ROOT}/include")
    set(MKL_LIBRARY_DIR "${MKL_ROOT}/lib/intel64")
    set(MKL_LIBRARIES mkl_rt pthread m dl)
    set(PIPER_HAVE_MKL ON)
  endif()
endif()

# =========================
# CPU 库
# =========================
file(GLOB CPU_SRC CONFIGURE_DEPENDS "src/*.cpp")
add_library(piperDNN STATIC ${CPU_SRC})

target_include_directories(piperDNN PUBLIC
  "${CMAKE_CURRENT_SOURCE_DIR}/src"
  "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

target_link_libraries(piperDNN PUBLIC
  Threads::Threads
  OpenMP::OpenMP_CXX
)

if(PIPER_HAVE_BLAS)
  target_link_libraries(piperDNN PUBLIC BLAS::BLAS)
  target_compile_definitions(piperDNN PUBLIC PIPER_HAVE_BLAS=1)
else()
  target_compile_definitions(piperDNN PUBLIC PIPER_HAVE_BLAS=0)
endif()

if(PIPER_HAVE_MKL)
  target_compile_definitions(piperDNN PUBLIC PIPER_HAVE_MKL=1)
else()
  target_compile_definitions(piperDNN PUBLIC PIPER_HAVE_MKL=0)
endif()

target_compile_options(piperDNN PRIVATE ${PIPER_OPT_FLAGS})

# =========================
# CUDA 库：按需启用
# =========================
set(PIPER_HAVE_CUDA OFF)
if(PIPER_ENABLE_CUDA)
  enable_language(CUDA)

  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  find_package(CUDAToolkit REQUIRED)

  file(GLOB CUDA_SRC CONFIGURE_DEPENDS "src/cuda/*.cu" "src/cuda/*.cuh")
  add_library(pipercuda STATIC ${CUDA_SRC})

  target_include_directories(pipercuda PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    ${CUDAToolkit_INCLUDE_DIRS}
  )

  target_link_libraries(pipercuda PUBLIC
    piperDNN
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
  )

  target_compile_options(pipercuda PRIVATE ${PIPER_OPT_FLAGS})

  set_target_properties(pipercuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  )

  set(PIPER_HAVE_CUDA ON)
endif()

# =========================
# 可执行文件：自动选择 main.cu / main.cpp
# =========================
set(MAIN_SOURCE "")
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu")
  if(PIPER_ENABLE_CUDA)
    set(MAIN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu")
  else()
    message(WARNING "src/main.cu exists but PIPER_ENABLE_CUDA=OFF, skipping executable.")
  endif()
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
  set(MAIN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
endif()

if(MAIN_SOURCE)
  add_executable(piper-DNN ${MAIN_SOURCE})

  target_link_libraries(piper-DNN PRIVATE piperDNN)

  if(PIPER_HAVE_CUDA)
    target_link_libraries(piper-DNN PRIVATE pipercuda CUDA::cudart)
    set_target_properties(piper-DNN PROPERTIES
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
  endif()

  target_compile_options(piper-DNN PRIVATE ${PIPER_OPT_FLAGS})
endif()

# =========================
# Tests
# =========================
if(PIPER_BUILD_TESTS)
  file(GLOB TEST_SOURCES_CPP "tests/*.cpp")
  file(GLOB TEST_SOURCES_CU  "tests/*.cu")

  # CUDA 未启用时，不编译 .cu 测试
  if(NOT PIPER_HAVE_CUDA)
    set(TEST_SOURCES ${TEST_SOURCES_CPP})
  else()
    set(TEST_SOURCES ${TEST_SOURCES_CPP} ${TEST_SOURCES_CU})
  endif()

  foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)

    # 依赖 CPU BLAS 的测试：没有就跳过
    if(test_name STREQUAL "test_gemm" AND NOT PIPER_HAVE_BLAS)
      continue()
    endif()

    # 依赖 MKL 的测试：没有就跳过
    if(test_name STREQUAL "test_spmv" AND NOT PIPER_HAVE_MKL)
      continue()
    endif()

    add_executable(${test_name} ${test_src})

    target_include_directories(${test_name} PRIVATE
      "${CMAKE_CURRENT_SOURCE_DIR}/src"
      "${CMAKE_CURRENT_SOURCE_DIR}/include"
    )

    target_link_libraries(${test_name} PRIVATE
      piperDNN
      OpenMP::OpenMP_CXX
      Threads::Threads
    )

    if(PIPER_HAVE_CUDA)
      target_include_directories(${test_name} PRIVATE ${CUDAToolkit_INCLUDE_DIRS})
      target_link_libraries(${test_name} PRIVATE
        pipercuda
        CUDA::cudart
        CUDA::cublas
      )
      set_target_properties(${test_name} PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      )
    endif()

    target_compile_options(${test_name} PRIVATE ${PIPER_OPT_FLAGS})

    if(test_name STREQUAL "test_spmv" AND PIPER_HAVE_MKL)
      target_include_directories(${test_name} PRIVATE "${MKL_INCLUDE_DIR}")
      target_link_directories(${test_name} PRIVATE "${MKL_LIBRARY_DIR}")
      target_link_libraries(${test_name} PRIVATE ${MKL_LIBRARIES})
    endif()

    if(test_name STREQUAL "test_gemm" AND PIPER_HAVE_BLAS)
      target_link_libraries(${test_name} PRIVATE BLAS::BLAS)
    endif()
  endforeach()
endif()

message(STATUS "==== PiperDNN Config Summary ====")
message(STATUS "PIPER_ENABLE_CUDA      = ${PIPER_ENABLE_CUDA}")
message(STATUS "PIPER_HAVE_CUDA        = ${PIPER_HAVE_CUDA}")
message(STATUS "PIPER_ENABLE_CPU_BLAS  = ${PIPER_ENABLE_CPU_BLAS}")
message(STATUS "PIPER_HAVE_BLAS        = ${PIPER_HAVE_BLAS}")
message(STATUS "PIPER_ENABLE_MKL       = ${PIPER_ENABLE_MKL}")
message(STATUS "PIPER_HAVE_MKL         = ${PIPER_HAVE_MKL}")
message(STATUS "=================================")
