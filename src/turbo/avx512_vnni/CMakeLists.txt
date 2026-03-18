include(${PROJECT_ROOT_DIR}/cmake/option.cmake)

if(NOT ANDROID AND AUTO_DETECT_ARCH)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|i686|i386|x64")
        # All .cc files under avx512_vnni/ share the same -march=avx512vnni flag.
        file(GLOB_RECURSE AVX512_VNNI_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/*.cc)
        set_source_files_properties(
            ${AVX512_VNNI_SRCS}
            PROPERTIES
            COMPILE_FLAGS "${TURBO_MARCH_FLAG_AVX512}"
        )
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
        message(STATUS "turbo/avx512_vnni: ARM64 detected, avx512_vnni sources will be compiled without special march flags")
    endif()
endif()
