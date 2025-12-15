cmake_minimum_required(VERSION 3.20)

# 先只声明 CXX，CUDA 由 PIPER_ENABLE_CUDA 控制按需启用
project(PiperDNN LANGUAGES CXX)

# =========================
# 基础配置
# =========================
# 仅对单配置生成器设置默认 build type（Ninja/Makefile）
if(NOT CMAKE_CONFIGURATION_TYPES AND NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
set(THREADS_PREFER_PTHREAD_FLAG TRUE)

# =========================
# 选项开关
# =========================
option(PIPER_ENABLE_CUDA       "Build CUDA backend (cuBLAS/cuSPARSE)" ON)
option(PIPER_BUILD_TESTS       "Build tests" ON)

option(PIPER_ENABLE_CPU_BLAS   "Enable CPU CBLAS provider (OpenBLAS / MKL)" OFF)
option(PIPER_REQUIRE_CPU_BLAS  "Fail if CPU CBLAS requested but missing" OFF)

option(PIPER_ENABLE_MKL        "Enable MKL detection (for MKL tests/paths)" OFF)

# GPU 架构（按需改）
set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86" CACHE STRING "CUDA architectures")

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
# MKL：按需检测（oneAPI）
# =========================
set(PIPER_HAVE_MKL OFF)

set(MKL_ROOT "" CACHE PATH "Intel oneAPI MKL root (contains include/ and lib/)")
if(PIPER_ENABLE_MKL)
  if(NOT MKL_ROOT AND DEFINED ENV{MKLROOT})
    set(MKL_ROOT "$ENV{MKLROOT}")
  endif()

  find_path(MKL_INCLUDE_DIR
    NAMES mkl.h
    HINTS "${MKL_ROOT}"
    PATH_SUFFIXES include
  )

  find_library(MKL_RT_LIB
    NAMES mkl_rt
    HINTS "${MKL_ROOT}"
    PATH_SUFFIXES lib lib/intel64 intel64 lib/x64
  )

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

# =========================
# CPU CBLAS：按需检测（优先 OpenBLAS；可手动指定）
# 你现在用的是 cblas_sgemm，所以必须是 CBLAS 提供者，不是纯 BLAS。
# =========================
set(PIPER_HAVE_BLAS OFF)

set(PIPER_OPENBLAS_ROOT "" CACHE PATH "OpenBLAS root (contains include/ and lib/)")
set(PIPER_OPENBLAS_LIB  "" CACHE FILEPATH "Path to OpenBLAS library (libopenblas.so / .dylib / .lib)")

set(PIPER_CBLAS_INCLUDE_DIR "")
set(PIPER_CBLAS_LIBRARIES "")

if(PIPER_ENABLE_CPU_BLAS)
  # 1) 优先：OpenBLAS（支持 cblas_*）
  if(NOT PIPER_OPENBLAS_LIB)
    find_library(PIPER_OPENBLAS_LIB
      NAMES openblas openblas64_ openblas64
      HINTS
        "${PIPER_OPENBLAS_ROOT}"
        "$ENV{OpenBLAS_HOME}" "$ENV{OPENBLAS_ROOT}"
        "$ENV{CONDA_PREFIX}"
      PATH_SUFFIXES lib lib64
    )
  endif()

  if(PIPER_OPENBLAS_LIB)
    # 头文件（可选，但跨平台更稳）
    find_path(PIPER_CBLAS_INCLUDE_DIR
      NAMES cblas.h
      HINTS
        "${PIPER_OPENBLAS_ROOT}"
        "$ENV{OpenBLAS_HOME}" "$ENV{OPENBLAS_ROOT}"
        "$ENV{CONDA_PREFIX}"
      PATH_SUFFIXES include include/openblas
    )
    set(PIPER_CBLAS_LIBRARIES "${PIPER_OPENBLAS_LIB}")
    set(PIPER_HAVE_BLAS ON)
  else()
    # 2) 兜底：如果 OpenBLAS 没找到，但 MKL 已找到，也可以当 CBLAS 提供者（mkl_rt 里也有 cblas_*）
    if(PIPER_HAVE_MKL)
      set(PIPER_CBLAS_LIBRARIES ${MKL_LIBRARIES})
      set(PIPER_CBLAS_INCLUDE_DIR "${MKL_INCLUDE_DIR}")
      set(PIPER_HAVE_BLAS ON)
    endif()
  endif()

  if(PIPER_HAVE_BLAS)
    add_library(Piper::CBLAS INTERFACE IMPORTED)
    if(PIPER_CBLAS_INCLUDE_DIR)
      target_include_directories(Piper::CBLAS INTERFACE "${PIPER_CBLAS_INCLUDE_DIR}")
    endif()
    target_link_libraries(Piper::CBLAS INTERFACE ${PIPER_CBLAS_LIBRARIES})
  elseif(PIPER_REQUIRE_CPU_BLAS)
    message(FATAL_ERROR
      "CPU CBLAS requested but no provider found.\n"
      " - Install OpenBLAS dev package, or\n"
      " - Set -DPIPER_OPENBLAS_LIB=/path/to/libopenblas.so, or\n"
      " - Enable MKL detection: -DPIPER_ENABLE_MKL=ON and set MKL_ROOT/MKLROOT."
    )
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
  target_link_libraries(piperDNN PUBLIC Piper::CBLAS)
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

  file(GLOB CUDA_SRC CONFIGURE_DEPENDS "src/cuda/*.cu")
  add_library(pipercuda STATIC ${CUDA_SRC})

  target_include_directories(pipercuda PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
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
  if(PIPER_HAVE_CUDA)
    set(MAIN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cu")
  else()
    message(WARNING "src/main.cu exists but CUDA is disabled, skipping piper-DNN.")
  endif()
elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
  set(MAIN_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")
endif()

if(MAIN_SOURCE)
  add_executable(piper-DNN ${MAIN_SOURCE})
  target_link_libraries(piper-DNN PRIVATE piperDNN)

  if(PIPER_HAVE_CUDA)
    target_link_libraries(piper-DNN PRIVATE pipercuda)
    set_target_properties(piper-DNN PROPERTIES
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
  endif()

  target_compile_options(piper-DNN PRIVATE ${PIPER_OPT_FLAGS})
endif()

# =========================
# Tests：tests 目录下每个源文件单独生成一个可执行
# =========================
if(PIPER_BUILD_TESTS)
  file(GLOB TEST_SOURCES_CPP CONFIGURE_DEPENDS "tests/*.cpp")
  file(GLOB TEST_SOURCES_CU  CONFIGURE_DEPENDS "tests/*.cu")

  set(TEST_SOURCES ${TEST_SOURCES_CPP})
  if(PIPER_HAVE_CUDA)
    list(APPEND TEST_SOURCES ${TEST_SOURCES_CU})
  endif()

  # 需要 CUDA 的测试（你现在的文件名）
  set(PIPER_TESTS_REQUIRE_CUDA
    test_axpy
    test_cuda_gemm
    test_cuBlas
    test_spmv
  )

  # 需要 CBLAS 的测试（会调用 cblas_*）
  set(PIPER_TESTS_REQUIRE_CBLAS
    test_cpu_gemm
    test_cuda_gemm
    test_cuBlas
  )

  # 需要 MKL 的测试（包含 <mkl.h>）
  set(PIPER_TESTS_REQUIRE_MKL
    test_spmv
  )

  foreach(test_src ${TEST_SOURCES})
    get_filename_component(test_name ${test_src} NAME_WE)

    # --- 跳过逻辑（在 add_executable 之前！） ---
    if(test_name IN_LIST PIPER_TESTS_REQUIRE_CUDA AND NOT PIPER_HAVE_CUDA)
      message(STATUS "Skipping ${test_name}: requires CUDA")
      continue()
    endif()

    if(test_name IN_LIST PIPER_TESTS_REQUIRE_CBLAS AND NOT PIPER_HAVE_BLAS)
      message(STATUS "Skipping ${test_name}: requires CPU CBLAS provider (OpenBLAS/MKL)")
      continue()
    endif()

    if(test_name IN_LIST PIPER_TESTS_REQUIRE_MKL AND NOT PIPER_HAVE_MKL)
      message(STATUS "Skipping ${test_name}: requires MKL (mkl.h)")
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

    # GPU 测试才链接 pipercuda（别让 CPU 测试无脑触发 CUDA device link）
    if(test_name IN_LIST PIPER_TESTS_REQUIRE_CUDA)
      target_link_libraries(${test_name} PRIVATE pipercuda)
      set_target_properties(${test_name} PROPERTIES
        CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
      )
    endif()

    if(test_name IN_LIST PIPER_TESTS_REQUIRE_MKL)
      target_link_libraries(${test_name} PRIVATE Piper::MKL)
    endif()

    target_compile_options(${test_name} PRIVATE ${PIPER_OPT_FLAGS})
  endforeach()
endif()

message(STATUS "==== PiperDNN Config Summary ====")
message(STATUS "PIPER_ENABLE_CUDA      = ${PIPER_ENABLE_CUDA}")
message(STATUS "PIPER_HAVE_CUDA        = ${PIPER_HAVE_CUDA}")
message(STATUS "PIPER_ENABLE_CPU_BLAS  = ${PIPER_ENABLE_CPU_BLAS}")
message(STATUS "PIPER_HAVE_BLAS        = ${PIPER_HAVE_BLAS}")
message(STATUS "PIPER_ENABLE_MKL       = ${PIPER_ENABLE_MKL}")
message(STATUS "PIPER_HAVE_MKL         = ${PIPER_HAVE_MKL}")
message(STATUS "OpenBLAS lib           = ${PIPER_OPENBLAS_LIB}")
message(STATUS "MKL_ROOT               = ${MKL_ROOT}")
message(STATUS "=================================")
