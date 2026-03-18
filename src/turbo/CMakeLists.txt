include(${PROJECT_ROOT_DIR}/cmake/bazel.cmake)
include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(NOT ANDROID AND AUTO_DETECT_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|i386|x64")
        setup_compiler_march_for_x86(TURBO_MARCH_FLAG_SSE TURBO_MARCH_FLAG_AVX2 TURBO_MARCH_FLAG_AVX512)
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/euclidean/avx512_impl.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/cosine/avx512_impl.cc
            ${CMAKE_CURRENT_SOURCE_DIR}/mips_euclidean/avx512_impl.cc
            PROPERTIES
            COMPILE_FLAGS "${TURBO_MARCH_FLAG_AVX512}"
        )
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        # ARM64 architecture - no special march flags needed for now
        # NEON implementations can be added here if needed
        message(STATUS "turbo: ARM64 detected, skipping x86-specific optimizations")
    endif()
endif()

file(GLOB_RECURSE ALL_SRCS *.cc *.c *.h)

cc_library(
    NAME zvec_turbo STATIC STRICT PACKED
    SRCS ${ALL_SRCS}
    INCS ${PROJECT_ROOT_DIR}/src/include/zvec
)
