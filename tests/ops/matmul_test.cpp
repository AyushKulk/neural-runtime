#include "runtime/ops.h"

#include <gtest/gtest.h>

// TODO: fill these in alongside implementing MatMul. Suggested cases:
//
//   - A known, hand-computed small case. E.g. A = {{1,2},{3,4}} (shape
//     {2,2}), B = {{5,6},{7,8}} (shape {2,2}) -> C should be
//     {{19,22},{43,50}}. Compute this by hand (or with numpy/pen and
//     paper) before writing the assertion — don't just trust whatever
//     the implementation produces.
//   - A non-square case, e.g. A is {2,3}, B is {3,2} -> output {2,2}.
//     Confirms the output shape formula, not just square-matrix math.
//   - Identity matrix: A x I == A, for some non-trivial A. Useful
//     property-based check that doesn't require hand-computing every
//     entry.
//   - (Once you're comfortable asserting on preconditions in a death
//     test) mismatched inner dimensions trigger the assert — see
//     GoogleTest's EXPECT_DEATH if you want to test that assert actually
//     fires, though this is optional polish, not required.
//
// TEST(MatMulTest, ComputesKnownSmallCase) { ... }

TEST(MatMulTest, ComputesKnownSmallCase) {
    runtime::Tensor A({2, 2});
    A.at({0, 0}) = 1.0f;
    A.at({0, 1}) = 2.0f;
    A.at({1, 0}) = 3.0f;
    A.at({1, 1}) = 4.0f;

    runtime::Tensor B({2, 2});
    B.at({0, 0}) = 5.0f;
    B.at({0, 1}) = 6.0f;
    B.at({1, 0}) = 7.0f;
    B.at({1, 1}) = 8.0f;

    runtime::Tensor C = runtime::MatMul(A, B);

    EXPECT_EQ(C.shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_EQ(C.at({0, 0}), 19.0f);
    EXPECT_EQ(C.at({0, 1}), 22.0f);
    EXPECT_EQ(C.at({1, 0}), 43.0f);
    EXPECT_EQ(C.at({1, 1}), 50.0f);
}

TEST(MatMulTest, ComputesNonSquareCase) {
    runtime::Tensor A({2, 3});
    A.at({0, 0}) = 1.0f;
    A.at({0, 1}) = 2.0f;
    A.at({0, 2}) = 3.0f;
    A.at({1, 0}) = 4.0f;
    A.at({1, 1}) = 5.0f;
    A.at({1, 2}) = 6.0f;

    runtime::Tensor B({3, 2});
    B.at({0, 0}) = 7.0f;
    B.at({0, 1}) = 8.0f;
    B.at({1, 0}) = 9.0f;
    B.at({1, 1}) = 10.0f;
    B.at({2, 0}) = 11.0f;
    B.at({2, 1}) = 12.0f;

    runtime::Tensor C = runtime::MatMul(A, B);

    EXPECT_EQ(C.shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_EQ(C.at({0, 0}), 58.0f);
    EXPECT_EQ(C.at({0, 1}), 64.0f);
    EXPECT_EQ(C.at({1, 0}), 139.0f);
    EXPECT_EQ(C.at({1, 1}), 154.0f);
}

TEST(MatMulTest, IdentityMatrix) {
    runtime::Tensor A({2, 2});
    A.at({0, 0}) = 1.0f;
    A.at({0, 1}) = 2.0f;
    A.at({1, 0}) = 3.0f;
    A.at({1, 1}) = 4.0f;

    runtime::Tensor I({2, 2});
    I.at({0, 0}) = 1.0f;
    I.at({0, 1}) = 0.0f;
    I.at({1, 0}) = 0.0f;
    I.at({1, 1}) = 1.0f;

    runtime::Tensor C = runtime::MatMul(A, I);

    EXPECT_EQ(C.shape(), (std::vector<std::size_t>{2, 2}));
    EXPECT_EQ(C.at({0, 0}), A.at({0, 0}));
    EXPECT_EQ(C.at({0, 1}), A.at({0, 1}));
    EXPECT_EQ(C.at({1, 0}), A.at({1, 0}));
    EXPECT_EQ(C.at({1, 1}), A.at({1, 1}));
}

TEST(MatMulTest, MismatchedInnerDimensionsTriggersAssert) {
    runtime::Tensor A({2, 3});
    runtime::Tensor B({4, 2});

    // This test is expected to trigger an assertion failure due to
    // mismatched inner dimensions (A's second dimension is 3, B's first
    // dimension is 4). Uncomment the following line to test the assertion.
    EXPECT_DEATH(runtime::MatMul(A, B), ".*");
}