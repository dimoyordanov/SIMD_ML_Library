cmake_minimum_required(VERSION 3.18)
project(ml_lib CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Release build with -O3 by default
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
add_compile_options($<$<CONFIG:Release>:-O3>)

# ------------------------------------------------------------------
# Locate ISPC compiler
# ------------------------------------------------------------------
find_program(ISPC_EXECUTABLE ispc
    HINTS
        "$ENV{HOME}/Downloads/ispc-trunk-linux/bin"
        "/usr/local/bin"
        "/usr/bin"
)
if(NOT ISPC_EXECUTABLE)
    message(FATAL_ERROR
        "ISPC compiler not found.\n"
        "Pass -DISPC_EXECUTABLE=/path/to/ispc or add it to PATH.")
endif()
message(STATUS "ISPC compiler: ${ISPC_EXECUTABLE}")

#neon-i16x16, neon-i16x8,  neon-i32x4, neon-i32x8, neon-i8x16, neon-i8x32
if(${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64|arm64")
    set(ISPC_TARGET "neon-i32x8" CACHE STRING
        "ISPC SIMD target architecture (e.g. avx, avx2, sse4)")
# avx512spr-x64
elseif(${CMAKE_SYSTEM_PROCESSOR} MATCHES "x86_64|amd64|x86")
    set(ISPC_TARGET "avx512spr-x32" CACHE STRING
        "ISPC SIMD target architecture (e.g. avx, avx2, sse4)")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

set(ISPC_GENERATED_DIR "${CMAKE_BINARY_DIR}/ispc_generated")
file(MAKE_DIRECTORY "${ISPC_GENERATED_DIR}")

# ------------------------------------------------------------------
# add_ispc_library(<name> <src1.ispc> [src2.ispc ...])
#
# Compiles each .ispc source with the ISPC compiler, collects the
# resulting object files into a static library, and makes the
# generated headers available via a public include directory.
# ------------------------------------------------------------------
function(add_ispc_library lib_name)
    set(ispc_objects)
    foreach(ispc_src IN LISTS ARGN)
        get_filename_component(base "${ispc_src}" NAME_WE)
        set(obj "${ISPC_GENERATED_DIR}/${base}.o")
        set(hdr "${ISPC_GENERATED_DIR}/${base}.h")

        add_custom_command(
            OUTPUT  "${obj}" "${hdr}"
            COMMAND "${ISPC_EXECUTABLE}"
                    --target=${ISPC_TARGET}
                    -O3
                    "${CMAKE_CURRENT_SOURCE_DIR}/${ispc_src}"
                    -o "${obj}"
                    -h "${hdr}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/${ispc_src}"
            COMMENT "ISPC: compiling ${ispc_src}"
            VERBATIM
        )
        list(APPEND ispc_objects "${obj}")
    endforeach()

    add_library("${lib_name}" STATIC ${ispc_objects})
    target_include_directories("${lib_name}" PUBLIC "${ISPC_GENERATED_DIR}")
    set_target_properties("${lib_name}" PROPERTIES LINKER_LANGUAGE CXX)
endfunction()

# ------------------------------------------------------------------
# ML kernels library
# ------------------------------------------------------------------
add_ispc_library(ml_kernels
    src/activation.ispc
    src/linear.ispc
    src/loss.ispc
    src/backprop.ispc
)

# ------------------------------------------------------------------
# C++ model-builder library
# ------------------------------------------------------------------
add_library(ml_model STATIC
    src/model.h
    src/model.cpp
    src/weight_io.h
    src/weight_io.cpp
)
target_include_directories(ml_model PUBLIC src)
target_link_libraries(ml_model PUBLIC ml_kernels)

# ------------------------------------------------------------------
# Executables
# ------------------------------------------------------------------
add_executable(benchmark  src/benchmark.cpp)
add_executable(run_model  src/run_model.cpp)

target_link_libraries(benchmark  PRIVATE ml_kernels)
target_link_libraries(run_model  PRIVATE ml_kernels)

add_executable(train_xor  example/train_xor.cpp)
target_link_libraries(train_xor PRIVATE ml_model)

# ------------------------------------------------------------------
# GoogleTest (mirrors Projects/Filesystem convention)
# ------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(tests)
