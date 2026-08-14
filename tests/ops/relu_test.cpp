#include "runtime/ops.h"

#include <gtest/gtest.h>

// TODO: fill these in alongside implementing Relu. Suggested cases:
//
//   - A tensor with a mix of negative, zero, and positive values -> only
//     the negatives should change (to 0); zero and positive pass
//     through unchanged. Cover all three in one test so a bug that only
//     breaks one branch doesn't slip through.
TEST(ReluTest, ClipsNegativesToZero) {
    runtime::Tensor input({2, 3});
    // Fill input with known values: some negative, some zero, some positive
    input.data()[0] = -1.0f;
    input.data()[1] = 0.0f;
    input.data()[2] = 1.0f;
    input.data()[3] = -2.0f;
    input.data()[4] = 2.0f;
    input.data()[5] = 3.0f;

    runtime::Tensor output = runtime::Relu(input);

    // Check every output element against the expected result
    for (std::size_t i = 0; i < output.numel(); ++i) {
        float expected = std::max(0.0f, input.data()[i]);
        EXPECT_EQ(output.data()[i], expected);
    }
}
//   - Output shape matches input shape.
TEST(ReluTest, OutputShapeMatchesInput) {
    runtime::Tensor input({4, 5});
    runtime::Tensor output = runtime::Relu(input);

    EXPECT_EQ(output.shape(), input.shape());
}
//
// TEST(ReluTest, ClipsNegativesToZero) { ... }


