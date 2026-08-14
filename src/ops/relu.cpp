#include "runtime/ops.h"

namespace runtime {

// Allocate an output the same shape as `input`, and set
// out[i] = max(0, input[i]) for every flat index i. Like Add's
// equal-shape case, a flat loop over numel() is fine — no need to go
// through multi-dimensional at().
Tensor Relu(const Tensor& input) {
    (void)input;

    Tensor output(input.shape());

    for (std::size_t i = 0; i < input.numel(); ++i) {
        output.data()[i] = std::max(0.0f, input.data()[i]);
    }

    return output;
}

}  // namespace runtime
