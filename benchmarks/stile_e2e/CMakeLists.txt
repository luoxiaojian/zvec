# Included into a zvec build through hook.cmake; builds the harness against
# the static libraries of that tree, so each zvec variant gets its own binary.
add_executable(zvec_e2e_bench "${CMAKE_CURRENT_LIST_DIR}/filter_bench.cc")
target_compile_features(zvec_e2e_bench PRIVATE cxx_std_17)
target_compile_options(zvec_e2e_bench PRIVATE -O3)
target_include_directories(zvec_e2e_bench PRIVATE
  "${PROJECT_BINARY_DIR}/src/generated" "${PROJECT_SOURCE_DIR}/src/include")
find_package(Threads REQUIRED)
# Factories register through static initializers: load whole archives.
set(_zvec_e2e_libs zvec zvec_core zvec_ailego zvec_turbo)
foreach(_lib IN LISTS _zvec_e2e_libs)
  if(APPLE)
    target_link_options(zvec_e2e_bench PRIVATE
      "-Wl,-force_load,$<TARGET_FILE:${_lib}>")
  else()
    target_link_options(zvec_e2e_bench PRIVATE
      "-Wl,--whole-archive,$<TARGET_FILE:${_lib}>,--no-whole-archive")
  endif()
endforeach()
target_link_libraries(zvec_e2e_bench PRIVATE ${_zvec_e2e_libs} Threads::Threads
  ${CMAKE_DL_LIBS})
