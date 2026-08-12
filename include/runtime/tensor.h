#pragma once

#include <cstddef>
#include <memory>
#include <vector>

#include "runtime/buffer.h"

namespace runtime {

// Tensor is shape/stride/offset metadata layered on top of a shared
// Buffer. It does not own memory directly — see buffer.h for why that
// split exists and what it costs (aliasing) vs. what it buys (cheap
// copies and, later, zero-copy views).
//
// Layout model (decided — see docs/design/tensor.md once you write it):
//   - float32 only for now, no dtype templating/erasure.
//   - Row-major (C order): for shape (d0, d1, ..., dn), a fully packed
//     contiguous tensor has strides (s0, s1, ..., sn) where
//         s[i] = product(d[i+1], d[i+2], ..., d[n])
//     and s[n] = 1. Concretely, shape {2, 3} -> strides {3, 1}.
//   - Every Tensor stores real strides + an element offset, even though
//     today every constructor only ever produces a contiguous tensor
//     (offset 0, strides = the formula above). This is intentionally
//     future-proofing for non-owning views (transpose, slice) without a
//     later redesign — don't skip populating strides/offset correctly
//     just because nothing reads them non-trivially yet.
//   - General indexing formula, given indices (i0, ..., in):
//         flat_offset = offset + i0*s0 + i1*s1 + ... + in*sn
//     This is the same address-generation math as indexing a
//     memory-mapped/DMA buffer from a stride/burst descriptor — the
//     concept transfers directly even though the syntax here is C++
//     rather than RTL.
//
// Alignment note (not needed yet): SIMD kernels later will want data
// aligned to 16/32/64 bytes. Buffer's default allocation strategy doesn't
// need to worry about this now — flagging it so a future change to
// Buffer's allocator isn't a surprise.
class Tensor {
public:
    // Allocates a new, contiguous, zero-offset Tensor with the given
    // shape (and a freshly-allocated Buffer to back it).
    explicit Tensor(const std::vector<std::size_t>& shape);

    // TODO: you'll likely also want a constructor that wraps an existing
    // shared_ptr<Buffer> + shape + strides + offset directly — that's
    // the constructor a future view/slice/transpose op would call. Worth
    // deciding its signature now even if nothing calls it yet.

    const std::vector<std::size_t>& shape() const;
    const std::vector<std::size_t>& strides() const;

    // Total element count (product of shape). Note: NOT the same as
    // Buffer::size() once views exist — a view can address fewer
    // elements than its underlying Buffer holds.
    std::size_t numel() const;

    // True iff strides exactly match the packed row-major formula above
    // for this tensor's shape (i.e. no gaps, no permutation). This is the
    // one invariant worth being able to check explicitly, since kernels
    // may want a fast contiguous path later.
    bool is_contiguous() const;

    // Element access. Bounds/rank checking should assert (or throw —
    // pick one deliberately and be consistent) rather than silently
    // reading out of bounds.
    float& at(const std::vector<std::size_t>& indices);
    const float& at(const std::vector<std::size_t>& indices) const;

    // Raw pointer to element 0 of *this tensor's view* (i.e. accounting
    // for offset_), not necessarily element 0 of the underlying Buffer.
    float* data();
    const float* data() const;

private:
    // TODO: helper to compute packed row-major strides from a shape —
    // both the constructor and is_contiguous() want this logic, so it
    // shouldn't live in either one directly. Free function or private
    // static method are both reasonable.

    std::shared_ptr<Buffer> buffer_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> strides_;
    std::size_t offset_ = 0;
};

}  // namespace runtime
