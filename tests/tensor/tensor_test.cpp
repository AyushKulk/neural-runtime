#include "runtime/tensor.h"

#include <gtest/gtest.h>

// TODO: fill these in as you implement Tensor. Suggested cases to cover —
// write the test before (or right alongside) the implementation it
// exercises, not after:
//
//   - Constructing a Tensor from a shape gives the expected shape() and
//     numel() (try a few shapes: {2,3}, {1}, {4,1,5}).

TEST(TensorTest, ConstructsWithExpectedShape) {
    runtime::Tensor t({2, 3});
    EXPECT_EQ(t.shape(), (std::vector<std::size_t>{2, 3}));
    EXPECT_EQ(t.numel(), 6);
    EXPECT_TRUE(t.is_contiguous());

    runtime::Tensor t2({1});
    EXPECT_EQ(t2.shape(), (std::vector<std::size_t>{1}));
    EXPECT_EQ(t2.numel(), 1);
    EXPECT_TRUE(t2.is_contiguous());

    runtime::Tensor t3({4, 1, 5});
    EXPECT_EQ(t3.shape(), (std::vector<std::size_t>{4, 1, 5}));
    EXPECT_EQ(t3.numel(), 20);
    EXPECT_TRUE(t3.is_contiguous());
}

//   - strides() matches the packed row-major formula for a given shape
//     (e.g. shape {2,3} -> strides {3,1}).
TEST(TensorTest, StridesMatchPackedRowMajor) {
    runtime::Tensor t({2, 3});
    EXPECT_EQ(t.strides(), (std::vector<std::size_t>{3, 1}));

    runtime::Tensor t2({4, 1, 5});
    EXPECT_EQ(t2.strides(), (std::vector<std::size_t>{5, 5, 1}));
}
//   - is_contiguous() is true right after construction.
TEST(TensorTest, IsContiguousAfterConstruction) {
    runtime::Tensor t({2, 3});
    EXPECT_TRUE(t.is_contiguous());

    runtime::Tensor t2({4, 1, 5});
    EXPECT_TRUE(t2.is_contiguous());
}
//   - at(indices) round-trips: write a value, read it back, at the same
//     indices.
TEST(TensorTest, AtRoundTrip) {
    runtime::Tensor t({2, 3});
    t.at({1, 2}) = 42.0f;
    EXPECT_EQ(t.at({1, 2}), 42.0f);

    runtime::Tensor t2({4, 1, 5});
    t2.at({3, 0, 4}) = -3.14f;
    EXPECT_EQ(t2.at({3, 0, 4}), -3.14f);
}
//   - at() addresses the correct flat offset — e.g. for shape {2,3},
//     at({1,0}) and at({0,0}) should differ by exactly stride[0] (=3)
//     elements in the underlying buffer. This is the test that would
//     catch a row-major/column-major mixup.
TEST(TensorTest, AtAddressesCorrectFlatOffset) {
    runtime::Tensor t({2, 3});
    t.at({0, 0}) = 1.0f;
    t.at({1, 0}) = 2.0f;
    EXPECT_EQ(t.at({1, 0}), 2.0f);
    EXPECT_EQ(t.at({0, 0}), 1.0f);
    EXPECT_EQ(t.strides()[0], 3); // stride[0] should be 3 for shape {2,3}
    // Compare addresses, not values: &t.at(...) is a float*, and pointer
    // subtraction gives element distance in the underlying buffer — the
    // values 1.0f/2.0f we wrote above are irrelevant to this check.
    EXPECT_EQ(&t.at({1, 0}) - &t.at({0, 0}),
              static_cast<std::ptrdiff_t>(t.strides()[0]));
}
//   - Two Tensors constructed independently do NOT share a Buffer
//     (writing through one doesn't affect the other) — this is the
//     baseline aliasing sanity check; a *copy* of a Tensor (once you
//     decide what Tensor's copy constructor does) is the case that
//     SHOULD share a Buffer, so it's worth a test either way once you've
//     decided.
TEST(TensorTest, IndependentTensorsDoNotShareBuffer) {
    runtime::Tensor t1({2, 3});
    runtime::Tensor t2({2, 3});
    t1.at({0, 0}) = 1.0f;
    t2.at({0, 0}) = 2.0f;
    EXPECT_EQ(t1.at({0, 0}), 1.0f);
    EXPECT_EQ(t2.at({0, 0}), 2.0f);
    EXPECT_NE(&t1.at({0, 0}), &t2.at({0, 0}));
}
// TEST(TensorTest, ConstructsWithExpectedShape) { ... }
