#include "runtime/buffer.h"

namespace runtime {

// Allocate storage for `count` floats and record the
// size. Decide the storage representation in buffer.h first (see the TODO
// there) — that choice determines what this constructor body looks like.
Buffer::Buffer(std::size_t count) : size_(count) {
    data_ = std::make_unique<float[]>(count);
    size_ = count;
}

// eturn a pointer to the underlying storage.
float* Buffer::data() { return data_.get(); }
// const version of the above.
const float* Buffer::data() const { return data_.get(); }

std::size_t Buffer::size() const { return size_; }

}  // namespace runtime
