file(GLOB_RECURSE PIPER_CUDA_SRCS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
)

if(NOT PIPER_CUDA_SRCS)
  message(FATAL_ERROR "No CUDA sources found under: ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

add_library(piper_rt_cuda STATIC ${PIPER_CUDA_SRCS})
add_library(piper-rt::cuda ALIAS piper_rt_cuda)

target_link_libraries(piper_rt_cuda
  PUBLIC
    piper-rt::core
    piper_rt_includes
    CUDA::cudart
    CUDA::cublas
    CUDA::cusparse
)

set_target_properties(piper_rt_cuda PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
)
