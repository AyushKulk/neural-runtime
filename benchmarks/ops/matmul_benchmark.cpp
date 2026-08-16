// Baseline MatMul benchmark. This measures the naive triple-loop
// implementation in src/ops/matmul.cpp EXACTLY AS IS — do not touch that
// file's loop order/structure to make this benchmark look better. This
// run's numbers become the "Baseline" row in the results table; every
// later optimization (tiled, threaded, fused) gets compared against
// THESE numbers, at THESE SAME sizes, or the comparison is meaningless.

#include <benchmark/benchmark.h>

#include "runtime/ops.h"
#include "runtime/tensor.h"

namespace {

// TODO: implement. Google Benchmark calls this function repeatedly,
// timing everything inside the `for (auto _ : state)` loop (setup before
// the loop, like constructing the input Tensors, is NOT timed — that's
// deliberate, you want to measure MatMul itself, not Tensor construction).
//
// Steps:
//   1. state.range(0) gives you the square matrix dimension for this
//      benchmark instance (wired up via the BENCHMARK(...)->Arg(...)
//      calls below) — call it n.
//   2. Construct two Tensors, shape {n, n}, filled with some values
//      (exact values don't matter for timing — arbitrary is fine, e.g.
//      fill with 1.0f or the flat index).
//   3. for (auto _ : state) { ... } — inside the loop, call
//      runtime::MatMul(a, b) and pass the result through
//      benchmark::DoNotOptimize(...) so the compiler can't prove the
//      result is unused and delete the whole call (see this milestone's
//      dead-code-elimination discussion — this line is the whole reason
//      Google Benchmark exists instead of hand-rolled chrono).
void BM_MatMul(benchmark::State& state) {
    
    int64_t n = state.range(0);
    // convert n to size_t for tensor dimensions
    size_t tensor_dim = static_cast<size_t>(n);
    runtime::Tensor a({tensor_dim, tensor_dim});
    runtime::Tensor b({tensor_dim, tensor_dim});

    // Fill the tensors with some values (e.g., 1.0f)
    for (size_t i = 0; i < tensor_dim; ++i) {
        for (size_t j = 0; j < tensor_dim; ++j) {
            a.at({i, j}) = 1.0f;
            b.at({i, j}) = 2.0f;
        }
    }

    for (auto _ : state) {
        runtime::Tensor result = runtime::MatMul(a, b);
        benchmark::DoNotOptimize(result);
    }
}

}  // namespace

// Args here are deliberately spread across a range: small enough to
// finish quickly (64), large enough that the matrices don't fit in L1/L2
// cache and tiling should visibly matter later (512). Revisit this range
// once you see the actual numbers — if 512 is too slow to iterate or 64
// is too noisy/dominated by overhead, that's real information, adjust.
BENCHMARK(BM_MatMul)->Arg(64)->Arg(128)->Arg(256)->Arg(512);
