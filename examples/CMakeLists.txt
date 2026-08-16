# Examples link against runtime_core exactly like the CLI does — nothing
# example-specific in the library itself, this is purely "assemble the
# pieces that already exist and demonstrate them working end to end."
add_executable(mlp_example mlp.cpp)

target_link_libraries(mlp_example
    PRIVATE
        runtime_core
        neural_runtime_warnings
)
