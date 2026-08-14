#pragma once

#include "runtime/tensor.h"

namespace runtime {

// Operators. Free functions (not a class hierarchy) — see the design
// discussion this milestone started from: a graph/Op-dispatch layer, if
// one gets built, can wrap these later without them needing to change.
// Each op takes const Tensor& inputs and returns a freshly allocated
// Tensor — no in-place/output-reuse variants yet (that's memory-reuse
// territory, deliberately deferred to the Week 2 systems milestone).
//
// Shape-contract violations (wrong rank, mismatched dims) are enforced
// with assert(), consistent with Tensor::at() — i.e. these are
// programmer-contract violations, not recoverable runtime errors, and
// they compile out in a Release/NDEBUG build. Revisit this once model
// (ONNX) input can violate a shape contract at runtime instead of only
// call-site code doing so.

// C = A x B. A must be {M, K}, B must be {K, N}, result is {M, N}.
// Both inputs must be rank 2 (no batching yet). This is intentionally the
// naive triple-nested-loop implementation — do not reorder the loops or
// otherwise optimize it here. It's meant to become the measured Week 2
// "Baseline" benchmark entry; optimized variants (tiled, threaded) should
// be separate functions/paths later, not a rewrite of this one in place.
Tensor MatMul(const Tensor& a, const Tensor& b);

// Elementwise add, with one restricted broadcasting case:
//   - a.shape() == b.shape(): ordinary elementwise add.
//   - a is {M, N} and b is {N} (rank 1): b is added to every row of a
//     (bias-add — the shape MatMul's output and a bias vector actually
//     have in the MatMul -> Add -> ReLU pipeline).
// No other broadcasting is supported — assert on any other shape
// combination rather than silently doing something unintended.
Tensor Add(const Tensor& a, const Tensor& b);

// Elementwise max(0, x). Rank/shape-agnostic — output shape always
// matches input shape.
Tensor Relu(const Tensor& input);

}  // namespace runtime
