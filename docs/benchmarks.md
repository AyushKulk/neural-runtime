# Benchmarks

Every number in this file is a real measured value copied from an actual
run of `neural_runtime_benchmarks` — never estimated, extrapolated, or
filled in ahead of actually running the suite. See the project's working
rules: optimizations are only claimed after baseline → measure → change →
measure → record.

## Methodology

- Benchmarks use [Google Benchmark](https://github.com/google/benchmark),
  fetched via CMake `FetchContent` (`benchmarks/CMakeLists.txt`) — same
  mechanism as GoogleTest.
- **Must** be built Release. `benchmarks/CMakeLists.txt` enforces this
  with a `FATAL_ERROR` at configure time if `CMAKE_BUILD_TYPE` isn't
  `Release` — a Debug build has asserts live and no optimizations, so its
  timings aren't representative of anything.
- Reproduce with:
  ```bash
  cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release -DNEURAL_RUNTIME_BUILD_BENCHMARKS=ON
  cmake --build build-release
  ./build-release/benchmarks/neural_runtime_benchmarks
  ```
- On macOS/Apple Silicon, Google Benchmark prints two benign warnings
  (`Unable to determine clock rate from sysctl: hw.cpufrequency` and
  `Failed to set thread affinity`) and a nonsense CPU frequency in the
  header (e.g. `10 X 24 MHz CPU s`). These are known platform
  metadata-detection gaps, not measurement errors — the `Time`/`CPU`
  columns come from the OS timer directly and are unaffected.
- What's measured in each `BM_*` function is the cost of the *whole* op
  call as the graph would actually invoke it — including the output
  Tensor's `Buffer` allocation (and, since the result is scoped inside
  the timed loop, its deallocation each iteration) — not just raw
  arithmetic. This is intentional: it's what `Graph::Execute()` actually
  pays per op, and it means the Week 2 memory-reuse work should show up
  in these same numbers later, not just loop-level optimizations.

## MatMul

### Baseline (naive triple-loop, `src/ops/matmul.cpp`)

Measured 2026-08-16, `build-release`, `benchmarks/ops/matmul_benchmark.cpp`.

| Size (n×n) | Time/iter    | CPU/iter     | Iterations |
|-----------:|-------------:|-------------:|-----------:|
|         64 |     8.92 ms  |     8.92 ms  |         72 |
|        128 |    70.38 ms  |    70.37 ms  |         10 |
|        256 |   558.77 ms  |   558.32 ms  |          1 |
|        512 |  4593.49 ms  |  4586.44 ms  |          1 |

**Scaling check**: naive MatMul is O(n³), so doubling `n` should ~8x the
time.

```
128/64  = 70.38 / 8.92   = 7.89x
256/128 = 558.77 / 70.38 = 7.94x
512/256 = 4593.49/558.77 = 8.22x
```

All within a few percent of the theoretical 8x — confirms the
implementation is genuinely O(n³) with no hidden complexity bug (e.g. an
accidental extra pass or unexpected allocation pattern would usually show
up as scaling that doesn't match the algorithm).

**Caveat**: at n=256 and n=512, Google Benchmark only ran 1 iteration
(already long enough to be stable by its own heuristic), so there's no
intra-run averaging to cushion against a one-off scheduling hiccup — the
machine's load average was ~2.4 at measurement time, not fully idle. If
tighter confidence is needed at n=512 specifically (the size where
tiling's cache benefit should matter most), rerun with
`->Repetitions(3)->ReportAggregatesOnly(true)` before treating it as
final.

### Results table

Filled in only after each optimization is implemented and its own
benchmark run completes — no placeholders standing in as if measured.

| Optimization      | n=64 | n=128 | n=256 | n=512 |
|-------------------|-----:|------:|------:|------:|
| Baseline          | 8.92 ms | 70.38 ms | 558.77 ms | 4593.49 ms |
| Tiled MatMul      | — | — | — | — |
| Multithreaded     | — | — | — | — |
| Fused operators   | — | — | — | — |
| Final             | — | — | — | — |

## Environment at time of measurement

- CPU: Apple Silicon, 10 cores (Google Benchmark's reported clock speed
  is wrong on this platform — see Methodology).
- Cache: L1 Data 64 KiB, L1 Instruction 128 KiB, L2 Unified 4096 KiB ×10.
- Build type: Release (`build-release/`).
- Toolchain: see [docs/environment.md](environment.md).
- Date: 2026-08-16.
