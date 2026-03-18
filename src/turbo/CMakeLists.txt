include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(NOT ANDROID AND AUTO_DETECT_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|i386|x64")
        setup_compiler_march_for_x86(TURBO_MARCH_FLAG_SSE TURBO_MARCH_FLAG_AVX2 TURBO_MARCH_FLAG_AVX512)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        # ARM64 architecture - no special march flags needed for now
        # NEON implementations can be added here if needed
        message(STATUS "turbo: ARM64 detected, skipping x86-specific optimizations")
    endif()
endif()

# Each ISA subdirectory manages its own per-file compile flags.
add_subdirectory(avx512_vnni)

file(GLOB_RECURSE ALL_SRCS *.cc *.c *.h)

cc_library(
    NAME zvec_turbo STATIC STRICT PACKED
    SRCS ${ALL_SRCS}
    LIBS zvec_ailego
    INCS ${CMAKE_CURRENT_SOURCE_DIR} ${PROJECT_ROOT_DIR}/src/include
)
