#include "runtime/ops.h"

#include <cassert>

namespace runtime {

//   1. assert both a and b are rank 2 (a.shape().size() == 2, same for b).
//   2. assert the inner dimensions match: a.shape()[1] == b.shape()[0].
//   3. Allocate the output Tensor with shape {a.shape()[0], b.shape()[1]}.
//   4. Fill it in with the naive i-j-k triple loop from ops.h's comment —
//      for each output element (i, j), accumulate the dot product of row
//      i of `a` and column j of `b`. Use a.at({i, k}) / b.at({k, j}) /
//      out.at({i, j}), or a.data()/b.data()/out.data() with manual index
//      math if you'd rather work with raw pointers directly (both are
//      fine here; raw pointers will matter more once you're optimizing
//      loop performance, not for a correctness-first baseline).
Tensor MatMul(const Tensor& a, const Tensor& b) {

    assert(a.shape().size() == 2 && b.shape().size() == 2);
    assert(a.shape()[1] == b.shape()[0]);

    std::size_t m = a.shape()[0];
    std::size_t n = a.shape()[1]; // same as b.shape()[0]
    std::size_t k = b.shape()[1];

    Tensor out({m, k});

    for (std::size_t i = 0; i < m; ++i) {
        for (std::size_t j = 0; j < k; ++j) {
            float sum = 0.0f;
            for (std::size_t p = 0; p < n; ++p) {
                sum += a.at({i, p}) * b.at({p, j});
            }
            out.at({i, j}) = sum;
        }
    }    
    return out;
}

}  // namespace runtime
