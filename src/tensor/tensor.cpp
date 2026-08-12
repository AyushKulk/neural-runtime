#include "runtime/tensor.h"

#include <cassert>
#include <numeric>

namespace runtime {

namespace {

// TODO: implement. Given a shape, compute the packed row-major strides
// (see the formula in tensor.h's class comment). Used by both the
// constructor below and is_contiguous().
std::vector<std::size_t> ComputeContiguousStrides(
    const std::vector<std::size_t>& shape) {    
    // e.g. shape is {3, 2}
    // explanation: the last dimension has stride 1,
    // the second-to-last dimension has stride equal 
    // to the size of the last dimension, and so on. So for shape {3, 2},
    // the strides would be {2, 1}.
    std::vector<std::size_t> strides(shape.size());
    std::size_t stride = 1;
    for (std::size_t i = shape.size(); i-->0;) {
        strides[i] = stride;
        stride *= shape[i];
    }
    (void)shape; // can this be removed? I think so, but let's keep it for now to avoid warnings.
    return strides;
}

}  // namespace
 
//   1. Compute contiguous strides for `shape`.
//   2. Figure out how many elements that is (product of shape — careful
//      with the empty-shape / scalar case if you want to support it).
//   3. Allocate a new Buffer of that size and take ownership via
//      shared_ptr.
//   4. offset_ stays 0 — this constructor always produces a fresh,
//      non-view tensor.
Tensor::Tensor(const std::vector<std::size_t>& shape) : shape_(shape) {
    strides_ = ComputeContiguousStrides(shape);
    std::size_t num_elements = 1;
    for (std::size_t dim : shape) {
        num_elements *= dim;
    }
    buffer_ = std::make_shared<Buffer>(num_elements);
}

const std::vector<std::size_t>& Tensor::shape() const { return shape_; }

const std::vector<std::size_t>& Tensor::strides() const { return strides_; }

// product of shape_.
std::size_t Tensor::numel() const { 
    return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<std::size_t>());
}

// compare strides_ against ComputeContiguousStrides(shape_).
bool Tensor::is_contiguous() const { 
    return strides_ == ComputeContiguousStrides(shape_);
 }


float& Tensor::at(const std::vector<std::size_t>& indices) {
    return const_cast<float&>(static_cast<const Tensor*>(this)->at(indices));
}

// Validate indices.size() == shape_.size() and each
// index is in bounds (assert, or throw — be consistent with the other
// bounds-checking in this class), then apply the offset + dot-product
// formula from tensor.h to find the flat index into buffer_->data().
const float& Tensor::at(const std::vector<std::size_t>& indices) const {
    assert(indices.size() == shape_.size()); // check rank
    for (std::size_t i = 0; i < indices.size(); ++i) {
        assert(indices[i] < shape_[i]); // check bounds
    }
    std::size_t flat_index = offset_ + std::inner_product(
        indices.begin(), indices.end(), strides_.begin(), std::size_t{0}); // offset + dot product of indices and strides
    return buffer_->data()[flat_index];
}

float* Tensor::data() { return buffer_->data() + offset_; }

const float* Tensor::data() const { return buffer_->data() + offset_; }

}  // namespace runtime
