# neural-runtime

A small neural network inference runtime built from scratch in C++20, for
the purpose of understanding and demonstrating how inference works
below the framework/API level: tensors, computation graphs, execution
engines, and optimized CPU kernels.

This is a portfolio project, developed incrementally in
milestones rather than generated all at once. It's intentionally much
smaller in scope than PyTorch, ONNX Runtime, or TensorRT.

## Status

**Milestone: environment + build scaffolding.** No tensor, op, or graph
code exists yet — see [docs/environment.md](docs/environment.md) for the
verified toolchain, and the "Planned" section below for what's next.

## Target architecture

```text
Model
  ↓
Model Parser
  ↓
Computation Graph
  ↓
Graph Optimizer
  ↓
Execution Engine
  ↓
Tensor / Memory System
  ↓
Optimized CPU Kernels
```

## Requirements

- C++20 compiler (Apple clang >= 17, or GCC >= 10 / Clang >= 12 on Linux)
- CMake >= 3.20
- A build tool: Ninja (preferred) or Make

GoogleTest is fetched automatically via CMake `FetchContent` — no manual
install required.

## Building

```bash
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## Running

```bash
./build/src/runtime
```

## Testing

```bash
ctest --test-dir build --output-on-failure
```

## Repository layout

```text
neural-runtime/
├── CMakeLists.txt      # Root build config
├── include/runtime/    # Public headers
├── src/
│   ├── runtime/         # Library entry points, version info
│   ├── tensor/          # (planned) tensor abstraction
│   ├── ops/             # (planned) operators: MatMul, Add, ReLU, ...
│   ├── graph/            # (planned) computation graph + optimizer
│   └── memory/           # (planned) memory management / reuse
├── tests/               # GoogleTest unit tests
├── benchmarks/          # (planned) benchmarking infrastructure
├── examples/            # (planned) example models/usage
├── models/              # (planned) sample model files
├── scripts/             # Python helpers: data gen, plotting, conversion
└── docs/                # Design notes, environment setup, architecture
```

## Planned

- **Week 1:** Tensor abstraction, MatMul/Add/ReLU operators, computation
  graph, execution, a small MLP running end to end, unit tests.
- **Week 2:** Multithreading, tiled MatMul, memory reuse, graph
  optimization, operator fusion, benchmarking + profiling.
- **Week 3:** ONNX model loading, CLI (`runtime run model.onnx`),
  integration tests, CI, architecture diagram, performance write-up.

See the project brief for full detail on each milestone.

## License

[MIT](LICENSE)
