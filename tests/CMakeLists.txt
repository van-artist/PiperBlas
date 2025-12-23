file(GLOB TEST_SOURCES_CPP CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
file(GLOB TEST_SOURCES_CU  CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cu")

# Special-gated tests (keep behavior consistent with your original)
set(TEST_SPMV_CPU "${CMAKE_CURRENT_SOURCE_DIR}/test_spmv_cpu.cpp")
set(TEST_CPU_GEMM "${CMAKE_CURRENT_SOURCE_DIR}/test_cpu_gemm.cpp")

list(REMOVE_ITEM TEST_SOURCES_CPP "${TEST_SPMV_CPU}")
list(REMOVE_ITEM TEST_SOURCES_CPP "${TEST_CPU_GEMM}")

# Generic C++ tests: link blas (+ cuda if available)
foreach(test_src ${TEST_SOURCES_CPP})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} ${test_src})
  target_link_libraries(${test_name} PRIVATE piper-rt::blas)
  if(PIPER_HAVE_CUDA AND TARGET piper_rt_cuda)
    target_link_libraries(${test_name} PRIVATE piper-rt::cuda)
  endif()
  add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()

# Eigen-gated test
if(PIPER_HAVE_EIGEN AND EXISTS "${TEST_SPMV_CPU}")
  add_executable(test_spmv_cpu "${TEST_SPMV_CPU}")
  target_link_libraries(test_spmv_cpu PRIVATE piper-rt::blas Eigen3::Eigen)
  if(PIPER_HAVE_CUDA AND TARGET piper_rt_cuda)
    target_link_libraries(test_spmv_cpu PRIVATE piper-rt::cuda)
  endif()
  add_test(NAME test_spmv_cpu COMMAND test_spmv_cpu)
else()
  message(STATUS "Skipping Eigen-dependent test: test_spmv_cpu")
endif()

# BLAS-gated test
if(PIPER_HAVE_BLAS AND EXISTS "${TEST_CPU_GEMM}")
  add_executable(test_cpu_gemm "${TEST_CPU_GEMM}")
  target_link_libraries(test_cpu_gemm PRIVATE piper-rt::blas Piper::CBLAS)
  add_test(NAME test_cpu_gemm COMMAND test_cpu_gemm)
else()
  message(STATUS "Skipping BLAS-dependent test: test_cpu_gemm")
endif()

# CUDA tests (.cu)
if(PIPER_HAVE_CUDA)
  foreach(test_src ${TEST_SOURCES_CU})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} PRIVATE piper-rt::blas piper-rt::cuda)
    set_target_properties(${test_name} PROPERTIES
      CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
    add_test(NAME ${test_name} COMMAND ${test_name})
  endforeach()
endif()
