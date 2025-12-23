file(GLOB_RECURSE PIPER_BLAS_SRCS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cxx"
)

# Optional MPI source: pi_summa.cpp (skip if no MPI)
if(NOT PIPER_HAVE_MPI)
  list(FILTER PIPER_BLAS_SRCS EXCLUDE REGEX ".*/pi_summa\\.(cpp|cc|cxx)$")
  message(STATUS "Skipping MPI-dependent source: src/blas/pi_summa.cpp")
endif()

if(NOT PIPER_BLAS_SRCS)
  message(FATAL_ERROR "No BLAS sources found under: ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

add_library(piper_rt_blas STATIC ${PIPER_BLAS_SRCS})
add_library(piper-rt::blas ALIAS piper_rt_blas)

target_link_libraries(piper_rt_blas PUBLIC piper-rt::core)

if(PIPER_HAVE_BLAS)
  target_link_libraries(piper_rt_blas PUBLIC Piper::CBLAS)
endif()
