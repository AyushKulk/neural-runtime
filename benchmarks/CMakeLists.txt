include(FetchContent)

# Same pattern as tests/CMakeLists.txt's GoogleTest fetch — same org,
# same mechanism, no system package required.
FetchContent_Declare(
    googlebenchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.1.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# Google Benchmark builds its own tests by default; we don't need them,
# and it pulls in GoogleTest again to do it if left on.
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

# Hard requirement, not a suggestion: benchmark numbers from a
# non-Release build are not real numbers (see the Debug-vs-Release
# discussion — no inlining/vectorization, asserts still live). Fail the
# configure step loudly rather than let a Debug-build benchmark result
# quietly end up in the results table.
if(NOT CMAKE_BUILD_TYPE STREQUAL "Release")
    message(FATAL_ERROR
        "Benchmarks must be built with -DCMAKE_BUILD_TYPE=Release "
        "(currently '${CMAKE_BUILD_TYPE}'). Use a separate build "
        "directory for benchmarking, e.g.:\n"
        "  cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release "
        "-DNEURAL_RUNTIME_BUILD_BENCHMARKS=ON")
endif()

add_executable(neural_runtime_benchmarks
    ops/matmul_benchmark.cpp
)

target_link_libraries(neural_runtime_benchmarks
    PRIVATE
        runtime_core
        neural_runtime_warnings
        benchmark::benchmark_main
)
