add_library(piper_rt_includes INTERFACE)
target_include_directories(piper_rt_includes INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
  $<INSTALL_INTERFACE:include>
)

add_subdirectory(core)
add_subdirectory(blas)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/dsp/CMakeLists.txt")
  add_subdirectory(dsp)
else()
  message(STATUS "src/dsp not found; skipping dsp module")
endif()

if(PIPER_HAVE_CUDA AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/CMakeLists.txt")
  add_subdirectory(cuda)
elseif(PIPER_HAVE_CUDA)
  message(STATUS "src/cuda not found; skipping cuda module")
endif()
