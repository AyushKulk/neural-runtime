#include "runtime/ops.h"

#include <cassert>

namespace runtime {

// Should handle exactly two cases (assert on anything
// else — see ops.h for the full contract):
//   1. a.shape() == b.shape(): allocate an output the same shape, and
//      out[i] = a[i] + b[i] for every flat index i (both sides are
//      contiguous right now, so a simple flat loop over numel() is fine
//      — you don't need to go through multi-dimensional at() here).
//   2. a is {M, N}, b is {N}: allocate output {M, N}, and for each row m
//      and column n, out.at({m, n}) = a.at({m, n}) + b.at({n}).
Tensor Add(const Tensor& a, const Tensor& b) {
    
    if (a.shape() == b.shape()) {
        Tensor output(a.shape());
        for (std::size_t i = 0; i < a.numel(); ++i) {
            output.data()[i] = a.data()[i] + b.data()[i];
        }
        return output;
    } else if (a.shape().size() == 2 && b.shape().size() == 1 &&
               a.shape()[1] == b.shape()[0]) {
        std::size_t m = a.shape()[0];
        std::size_t n = a.shape()[1];
        Tensor output({m, n});
        for (std::size_t i = 0; i < m; ++i) {
            for (std::size_t j = 0; j < n; ++j) {
                output.at({i, j}) = a.at({i, j}) + b.at({j});
            }
        }
        return output;
    } else {
        assert(false && "Unsupported shape combination for Add");
    }
    return Tensor({0}); // This line will never be reached due to the assert, but is needed to satisfy the compiler.
}

}  // namespace runtime
