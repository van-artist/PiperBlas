cmake_minimum_required(VERSION 3.20)

project(PiperRT LANGUAGES CXX)

# Default build type for single-config generators
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

option(PIPER_BUILD_TESTS        "Build tests" ON)
option(PIPER_ENABLE_MKL         "Enable MKL detection (for MKL tests/paths)" ON)
option(PIPER_ENABLE_EIGEN       "Enable Eigen support (for Eigen-based tests)" ON)
option(PIPER_ENABLE_MPI         "Enable MPI support" ON)
option(PIPER_ENABLE_CPU_BLAS    "Enable CPU BLAS support" ON)
option(PIPER_ENABLE_CUDA        "Enable CUDA support" ON)

set(CMAKE_CUDA_ARCHITECTURES "70;75;80;86" CACHE STRING "CUDA architectures")

# ---- Common build interface target (propagates to all targets that link it) ----
find_package(Threads REQUIRED)

add_library(piper_rt_build INTERFACE)
target_link_libraries(piper_rt_build INTERFACE Threads::Threads)
target_compile_features(piper_rt_build INTERFACE cxx_std_17)

set(PIPER_OPT_FLAGS
  $<$<AND:$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>,$<CONFIG:Release>>:-O3 -ffast-math -march=native>
  $<$<AND:$<CXX_COMPILER_ID:AppleClang>,$<CONFIG:Release>>:-O3 -ffast-math -mcpu=native>
  $<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2>
)
target_compile_options(piper_rt_build INTERFACE ${PIPER_OPT_FLAGS})

# ---- Dependency detection -> PIPER_HAVE_* ----
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
    add_library(Piper::MKL INTERFACE IMPORTED)
    target_include_directories(Piper::MKL INTERFACE "${MKL_INCLUDE_DIR}")

    set(MKL_LIBRARIES "${MKL_RT_LIB}")
    if(UNIX AND NOT APPLE)
      list(APPEND MKL_LIBRARIES pthread m dl)
    endif()
    target_link_libraries(Piper::MKL INTERFACE ${MKL_LIBRARIES})
  else()
    message(STATUS "MKL not found (MKL_ROOT='${MKL_ROOT}'); MKL-dependent components will be skipped")
  endif()
endif()

# BLAS / CBLAS
set(PIPER_HAVE_BLAS OFF)
set(PIPER_OPENBLAS_ROOT "" CACHE PATH "Optional: OpenBLAS installation root (for locating cblas.h)")
set(PIPER_CBLAS_INCLUDE_DIR "" CACHE PATH "Optional: CBLAS include directory override (contains cblas.h)")

if(PIPER_ENABLE_CPU_BLAS)
  find_package(BLAS QUIET)
  if(BLAS_FOUND)
    # Provide a stable target regardless of whether BLAS::BLAS exists
    add_library(Piper::BLAS INTERFACE IMPORTED)
    if(TARGET BLAS::BLAS)
      target_link_libraries(Piper::BLAS INTERFACE BLAS::BLAS)
    else()
      target_link_libraries(Piper::BLAS INTERFACE ${BLAS_LIBRARIES})
    endif()

    if(NOT PIPER_CBLAS_INCLUDE_DIR)
      find_path(PIPER_CBLAS_INCLUDE_DIR
        NAMES cblas.h
        HINTS
          "${PIPER_OPENBLAS_ROOT}"
          "$ENV{OpenBLAS_HOME}" "$ENV{OPENBLAS_ROOT}" "$ENV{CONDA_PREFIX}" "/usr"
        PATH_SUFFIXES
          include include/openblas include/x86_64-linux-gnu include/x86_64-linux-gnu/openblas-pthread
      )
    endif()

    add_library(Piper::CBLAS INTERFACE IMPORTED)
    if(PIPER_CBLAS_INCLUDE_DIR)
      target_include_directories(Piper::CBLAS INTERFACE "${PIPER_CBLAS_INCLUDE_DIR}")
    endif()
    target_link_libraries(Piper::CBLAS INTERFACE Piper::BLAS)

    set(PIPER_HAVE_BLAS ON)
  else()
    message(STATUS "BLAS not found; BLAS-dependent components will be skipped")
  endif()
else()
  message(STATUS "BLAS support disabled via PIPER_ENABLE_CPU_BLAS=OFF")
endif()

# CUDA (truly optional)
set(PIPER_HAVE_CUDA OFF)
if(PIPER_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  find_package(CUDAToolkit REQUIRED)
  set(PIPER_HAVE_CUDA ON)
else()
  message(STATUS "CUDA support disabled via PIPER_ENABLE_CUDA=OFF")
endif()

# ---- Config/feature macros propagated via one interface target ----
add_library(piper_rt_config INTERFACE)
target_compile_definitions(piper_rt_config INTERFACE
  PIPER_HAVE_BLAS=$<BOOL:${PIPER_HAVE_BLAS}>
  PIPER_HAVE_MKL=$<BOOL:${PIPER_HAVE_MKL}>
  PIPER_HAVE_MPI=$<BOOL:${PIPER_HAVE_MPI}>
  PIPER_HAVE_CUDA=$<BOOL:${PIPER_HAVE_CUDA}>
  PIPER_HAVE_EIGEN=$<BOOL:${PIPER_HAVE_EIGEN}>
)

# ---- Build tree ----
add_subdirectory(src)

if(PIPER_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

add_subdirectory(bin)

# ---- Summary ----
message(STATUS "==== PiperRT Config Summary ====")
message(STATUS "PIPER_BUILD_TESTS       = ${PIPER_BUILD_TESTS}")
message(STATUS "PIPER_ENABLE_CUDA       = ${PIPER_ENABLE_CUDA}")
message(STATUS "PIPER_HAVE_CUDA         = ${PIPER_HAVE_CUDA}")
message(STATUS "CMAKE_CUDA_ARCHITECTURES= ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "PIPER_ENABLE_CPU_BLAS   = ${PIPER_ENABLE_CPU_BLAS}")
message(STATUS "PIPER_HAVE_BLAS         = ${PIPER_HAVE_BLAS}")
message(STATUS "PIPER_OPENBLAS_ROOT     = ${PIPER_OPENBLAS_ROOT}")
message(STATUS "PIPER_CBLAS_INCLUDE_DIR = ${PIPER_CBLAS_INCLUDE_DIR}")
message(STATUS "BLAS_LIBRARIES          = ${BLAS_LIBRARIES}")
message(STATUS "PIPER_ENABLE_MKL        = ${PIPER_ENABLE_MKL}")
message(STATUS "PIPER_HAVE_MKL          = ${PIPER_HAVE_MKL}")
message(STATUS "MKL_ROOT                = ${MKL_ROOT}")
message(STATUS "PIPER_ENABLE_MPI        = ${PIPER_ENABLE_MPI}")
message(STATUS "PIPER_HAVE_MPI          = ${PIPER_HAVE_MPI}")
message(STATUS "PIPER_ENABLE_EIGEN      = ${PIPER_ENABLE_EIGEN}")
message(STATUS "PIPER_HAVE_EIGEN        = ${PIPER_HAVE_EIGEN}")
message(STATUS "================================")
