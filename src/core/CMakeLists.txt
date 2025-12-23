file(GLOB_RECURSE PIPER_CORE_SRCS CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cxx"
)

if(NOT PIPER_CORE_SRCS)
  message(FATAL_ERROR "No core sources found under: ${CMAKE_CURRENT_SOURCE_DIR}")
endif()

add_library(piper_rt_core STATIC ${PIPER_CORE_SRCS})
add_library(piper-rt::core ALIAS piper_rt_core)

target_link_libraries(piper_rt_core
  PUBLIC
    piper_rt_build
    piper_rt_config
    piper_rt_includes
)

if(PIPER_HAVE_CUDA)
  target_link_libraries(piper_rt_core PUBLIC CUDA::cudart)
endif()
