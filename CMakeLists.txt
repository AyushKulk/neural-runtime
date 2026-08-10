cmake_minimum_required(VERSION 3.20)

project(neural_runtime
    VERSION 0.1.0
    DESCRIPTION "A small neural network inference runtime built from scratch"
    LANGUAGES CXX
)

# ---------------------------------------------------------------------------
# Project-wide settings
# ---------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Always emit compile_commands.json for editor tooling (clangd, etc).
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to a Release build if the user didn't ask for anything specific.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type" FORCE)
endif()

option(NEURAL_RUNTIME_BUILD_TESTS "Build unit tests" ON)
option(NEURAL_RUNTIME_BUILD_BENCHMARKS "Build benchmarks" OFF)
option(NEURAL_RUNTIME_WARNINGS_AS_ERRORS "Treat compiler warnings as errors" OFF)

# ---------------------------------------------------------------------------
# Common compiler warnings, applied via an INTERFACE target so every other
# target in the project can opt in with target_link_libraries(... PRIVATE
# neural_runtime_warnings) instead of repeating flags everywhere.
# ---------------------------------------------------------------------------

add_library(neural_runtime_warnings INTERFACE)
target_compile_options(neural_runtime_warnings INTERFACE
    -Wall
    -Wextra
    -Wpedantic
    -Wshadow
    -Wconversion
    -Wsign-conversion
)
if(NEURAL_RUNTIME_WARNINGS_AS_ERRORS)
    target_compile_options(neural_runtime_warnings INTERFACE -Werror)
endif()

# ---------------------------------------------------------------------------
# Subdirectories
# ---------------------------------------------------------------------------

add_subdirectory(src)

if(NEURAL_RUNTIME_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(NEURAL_RUNTIME_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
