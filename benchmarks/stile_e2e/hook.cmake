# cmake -DCMAKE_PROJECT_zvec_INCLUDE=/Users/luo/workspace/zvec-e2e/hook.cmake
# Adds the harness after the zvec root has defined its SDK targets. The path
# is baked in now: deferred calls expand variables when they run.
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  cmake_language(EVAL CODE
    "cmake_language(DEFER DIRECTORY \"${CMAKE_SOURCE_DIR}\" CALL include \"${CMAKE_CURRENT_LIST_DIR}/CMakeLists.txt\")")
endif()
