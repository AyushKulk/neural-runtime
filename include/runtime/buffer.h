#pragma once

#include <cstddef>
#include <memory>

namespace runtime {

// Buffer owns a flat, contiguous block of `float`s and nothing else — no
// shape, no strides, no notion of "tensor". Its only job is memory
// lifetime: allocate N floats, and free them exactly once, whenever the
// last owner goes away.
//
// This is deliberately the *only* thing in the codebase that calls new/
// delete (or an allocator) for tensor data. Tensor (see tensor.h) holds a
// shared_ptr<Buffer> plus shape/stride/offset metadata layered on top —
// multiple Tensors can point at the same Buffer (e.g. future views/slices)
// without copying data, at the cost of needing to reason about aliasing
// once ops start writing in place.
//
// Design notes / things to get right when implementing this:
//   - RAII: whatever you allocate with, make sure it's freed exactly once,
//     with no manual delete[] calls scattered elsewhere. shared_ptr's
//     custom deleter (or std::vector, or std::unique_ptr<float[]>) should
//     be the *only* place that knows how the memory was obtained.
//   - Move vs copy: think about whether Buffer itself should be copyable
//     at all. Copying a Buffer (deep data copy) is a different operation
//     from copying a Tensor (shared_ptr refcount bump) — make sure it's
//     obvious at the call site which one is happening.
//   - const correctness: a `const Buffer&` should only be able to read
//     data, never write through it.
class Buffer {
public:
    // Allocates storage for `count` floats. Contents are left as
    // implementation-defined for now (decide: zero-initialize, or leave
    // uninitialized for performance and require callers to fill it?).
    explicit Buffer(std::size_t count);

    // TODO: decide Buffer's copy/move semantics deliberately, don't just
    // accept whatever the compiler defaults to. Given shared_ptr<Buffer>
    // is how Tensor shares memory, does Buffer itself ever need to be
    // copied directly?
    std::unique_ptr<Buffer> clone() const;

    // Raw access to the underlying storage. Non-const overload lets
    // callers write; const overload doesn't.
    float* data();
    const float* data() const;

    std::size_t size() const;

private:
    // TODO: pick the underlying storage representation. Common options:
    //   - std::unique_ptr<float[]>
    //   - std::vector<float>
    //   - a raw pointer you manage yourself (only if you have a specific
    //     reason, e.g. wanting to control alignment later — not needed
    //     yet, see the alignment note in tensor.h)
    std::unique_ptr<float[]> data_;
    std::size_t size_;
};

}  // namespace runtime
