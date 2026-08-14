# runtime_core: the library that will eventually host the tensor, ops,
# graph, memory, and execution-engine code. Kept as a real library target
# (rather than folding everything into the CLI executable) so tests can link
# against it directly without going through main().
add_library(runtime_core
    runtime/version.cpp
    tensor/buffer.cpp
    tensor/tensor.cpp
    ops/matmul.cpp
    ops/add.cpp
    ops/relu.cpp
)

target_include_directories(runtime_core
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

target_link_libraries(runtime_core PRIVATE neural_runtime_warnings)

# CLI executable. `runtime run model.onnx` etc. will grow here in Week 3;
# for now it just prints version info to confirm the build works.
add_executable(neural_runtime_cli main.cpp)
set_target_properties(neural_runtime_cli PROPERTIES OUTPUT_NAME "runtime")

target_link_libraries(neural_runtime_cli
    PRIVATE
        runtime_core
        neural_runtime_warnings
)
