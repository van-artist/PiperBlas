# Collect bin sources (non-recursive; add RECURSE only if你bin下面还有子目录)
file(GLOB BIN_SOURCES_CPP CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/*.cxx"
)

set(BIN_SOURCES_CU "")
if(PIPER_HAVE_CUDA)
  file(GLOB BIN_SOURCES_CU CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cu"
  )
endif()

message(STATUS "bin/: cpp sources = ${BIN_SOURCES_CPP}")
message(STATUS "bin/: cu sources  = ${BIN_SOURCES_CU}")

# If you expect sources but none found, fail with a useful hint.
if(NOT BIN_SOURCES_CPP AND NOT BIN_SOURCES_CU)
  file(GLOB _BIN_ALL RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*")
  message(FATAL_ERROR
    "No bin sources matched patterns under: ${CMAKE_CURRENT_SOURCE_DIR}\n"
    "Looked for: *.cpp *.cc *.cxx"
    "$<$<BOOL:${PIPER_HAVE_CUDA}>: *.cu>\n"
    "Directory contents: ${_BIN_ALL}\n"
    "If your files are in subdirectories, switch to GLOB_RECURSE."
  )
endif()

# Build cpp binaries
foreach(bin_src ${BIN_SOURCES_CPP})
  get_filename_component(bin_name ${bin_src} NAME_WE)
  add_executable(${bin_name} ${bin_src})

  # Prefer blas if available; otherwise core
  if(TARGET piper_rt_blas)
    target_link_libraries(${bin_name} PRIVATE piper-rt::blas)
  else()
    target_link_libraries(${bin_name} PRIVATE piper-rt::core)
  endif()

  if(PIPER_HAVE_CUDA AND TARGET piper_rt_cuda)
    target_link_libraries(${bin_name} PRIVATE piper-rt::cuda)
  endif()
endforeach()

# Build cu binaries
foreach(bin_src ${BIN_SOURCES_CU})
  get_filename_component(bin_name ${bin_src} NAME_WE)
  add_executable(${bin_name} ${bin_src})

  if(TARGET piper_rt_blas)
    target_link_libraries(${bin_name} PRIVATE piper-rt::blas)
  else()
    target_link_libraries(${bin_name} PRIVATE piper-rt::core)
  endif()

  if(TARGET piper_rt_cuda)
    target_link_libraries(${bin_name} PRIVATE piper-rt::cuda)
    set_target_properties(${bin_name} PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
  endif()
endforeach()
