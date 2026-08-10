include(FetchContent)

# Pin GoogleTest to a specific release for reproducible builds. Fetched
# on-demand rather than requiring a system package, so `cmake --build`
# works the same way on a fresh macOS or Linux machine.
FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.tar.gz
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
# Avoid overriding the parent project's compiler/linker settings.
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(neural_runtime_tests
    runtime/version_test.cpp
)

target_link_libraries(neural_runtime_tests
    PRIVATE
        runtime_core
        neural_runtime_warnings
        GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(neural_runtime_tests)
