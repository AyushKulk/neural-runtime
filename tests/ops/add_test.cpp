#include "runtime/ops.h"

#include <gtest/gtest.h>

// TODO: fill these in alongside implementing Add. Suggested cases:
//
//   - Equal-shape add: two {2,3} tensors with known values, check every
//     output element against a hand-computed sum.

TEST(AddTest, ComputesEqualShapeSum) {
    runtime::Tensor a({2, 3});
    runtime::Tensor b({2, 3});

    // Fill a and b with known values
    for (std::size_t i = 0; i < a.numel(); ++i) {
        a.data()[i] = static_cast<float>(i);
        b.data()[i] = static_cast<float>(i * 2);
    }

    runtime::Tensor out = runtime::Add(a, b);

    // Check every output element against the expected sum
    for (std::size_t i = 0; i < out.numel(); ++i) {
        EXPECT_EQ(out.data()[i], a.data()[i] + b.data()[i]);
    }
}

//   - Bias-add broadcast: a {2,3} tensor plus a {3} tensor, check that
//     the {3} vector was added to *every row*, not just row 0 — this is
//     the case most likely to have an off-by-one/wrong-loop bug, so
//     don't just check one row and assume the rest are right.
TEST(AddTest, ComputesBiasAddBroadcast) {
    runtime::Tensor a({2, 3});
    runtime::Tensor b({3});

    // Fill a with known values
    for (std::size_t i = 0; i < a.numel(); ++i) {
        a.data()[i] = static_cast<float>(i);
    }

    // Fill b with known values
    for (std::size_t i = 0; i < b.numel(); ++i) {
        b.data()[i] = static_cast<float>(i * 10);
    }

    runtime::Tensor out = runtime::Add(a, b);

    // Check every output element against the expected sum
    for (std::size_t i = 0; i < out.shape()[0]; ++i) { // iterate over rows
        for (std::size_t j = 0; j < out.shape()[1]; ++j) { // iterate over columns
            EXPECT_EQ(out.at({i, j}), a.at({i, j}) + b.at({j}));
        }
    }
}

//   - (Optional) confirm a shape combination outside the two supported
//     cases actually asserts, if you want to exercise that path.
TEST(AddTest, RejectsIncompatibleShapes) {
    runtime::Tensor a({2, 3});
    runtime::Tensor b({3, 2});

    EXPECT_DEATH(runtime::Add(a, b), ".*"); // Expect the program to terminate due to assertion failure
}
