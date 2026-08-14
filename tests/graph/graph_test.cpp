#include "runtime/graph.h"

#include <gtest/gtest.h>

// TODO: fill these in alongside implementing Graph. Suggested cases,
// roughly in the order they build on each other:
//
//   - Single Input node, no ops: Execute() with a value for that input,
//     output_id = the input's own node ID, should just return that value
//     back unchanged. Simplest possible thing that could work — good
//     first test to write since it doesn't exercise topological sort or
//     any op dispatch at all, just Input plumbing.
TEST(GraphTest, SingleInputReturnsItsOwnValue) {
    runtime::Graph g;
    std::size_t input_id = g.AddInput("input");
    runtime::Tensor input_tensor({2, 2});
    for (std::size_t i = 0; i < input_tensor.numel(); ++i) {
        input_tensor.data()[i] = 1.0f;
    }
    std::unordered_map<std::size_t, runtime::Tensor> inputs = {{input_id, input_tensor}};
    runtime::Tensor output_tensor = g.Execute(inputs, input_id);
    EXPECT_EQ(output_tensor.numel(), input_tensor.numel());
    for (std::size_t i = 0; i < output_tensor.numel(); ++i) {
        EXPECT_FLOAT_EQ(output_tensor.data()[i], input_tensor.data()[i]);
    }
}
//   - Single Relu node fed by a single Input: confirms one level of
//     dependency executes correctly (Input's value flows into the Relu
//     node's computation).
TEST(GraphTest, SingleReluNode) {
    runtime::Graph g;
    std::size_t input_id = g.AddInput("input");
    std::size_t relu_id = g.AddOp(runtime::OpType::kRelu, {input_id}, "relu");
    runtime::Tensor input_tensor({2, 2});
    for (std::size_t i = 0; i < input_tensor.numel(); ++i) {
        input_tensor.data()[i] = -1.0f;
    }
    std::unordered_map<std::size_t, runtime::Tensor> inputs = {{input_id, input_tensor}};
    runtime::Tensor output_tensor = g.Execute(inputs, relu_id);
    EXPECT_EQ(output_tensor.numel(), input_tensor.numel());
    for (std::size_t i = 0; i < output_tensor.numel(); ++i) {
        EXPECT_FLOAT_EQ(output_tensor.data()[i], 0.0f);  // Relu of negative is 0
    }
}
//   - The actual milestone-one pipeline: Input -> MatMul -> Add -> Relu
//     -> Output. Build it with a Constant weight matrix and a Constant
//     bias vector (AddConstant), one AddInput for the runtime input, and
//     three AddOp calls (kMatMul, kAdd, kRelu) chained via input_ids.
//     Execute() and check the result against a hand-computed expected
//     Tensor (you already know how to do this from the MatMul tests).
TEST(GraphTest, FullPipeline) {
    runtime::Graph g;
    std::size_t input_id = g.AddInput("input");
    runtime::Tensor weight_tensor({2, 2});
    for (std::size_t i = 0; i < weight_tensor.numel(); ++i) {
        weight_tensor.data()[i] = 2.0f;
    }
    std::size_t weight_id = g.AddConstant(weight_tensor, "weight");
    runtime::Tensor bias_tensor({2});
    for (std::size_t i = 0; i < bias_tensor.numel(); ++i) {
        bias_tensor.data()[i] = 1.0f;
    }
    std::size_t bias_id = g.AddConstant(bias_tensor, "bias");
    std::size_t matmul_id = g.AddOp(runtime::OpType::kMatMul, {input_id, weight_id}, "matmul");
    std::size_t add_id = g.AddOp(runtime::OpType::kAdd, {matmul_id, bias_id}, "add");
    std::size_t relu_id = g.AddOp(runtime::OpType::kRelu, {add_id}, "relu");

    // Rank 2, not rank 1 — MatMul requires both operands rank 2 (see
    // ops.h's contract). Shape {1, 2}: one "example" with 2 features.
    runtime::Tensor input_tensor({1, 2});
    for (std::size_t i = 0; i < input_tensor.numel(); ++i) {
        input_tensor.data()[i] = 3.0f;
    }
    std::unordered_map<std::size_t, runtime::Tensor> inputs = {{input_id, input_tensor}};
    runtime::Tensor output_tensor = g.Execute(inputs, relu_id);

    // Expected output: Relu((input @ weight) + bias).
    // input @ weight: each output entry sums over the shared dimension
    // (real matrix multiplication, not elementwise) — with input=[3,3]
    // and every weight entry 2.0, each entry is 3*2 + 3*2 = 12.
    // + bias[1,1] = [13,13]. Relu leaves it unchanged (already positive).
    EXPECT_EQ(output_tensor.numel(), 2u);
    for (std::size_t i = 0; i < output_tensor.numel(); ++i) {
        EXPECT_FLOAT_EQ(output_tensor.data()[i], 13.0f);
    }
}
//   - Executing the same Graph twice with different Input values should
//     give different, independently-correct outputs — this is the test
//     that would catch a bug where a node's stale `value` from a
//     previous Execute() call leaks into a later one instead of being
//     recomputed.
TEST(GraphTest, ExecuteTwiceWithDifferentInputsGivesIndependentResults) {
    runtime::Graph g;
    std::size_t input_id = g.AddInput("input");
    runtime::Tensor weight_tensor({2, 2});
    for (std::size_t i = 0; i < weight_tensor.numel(); ++i) {
        weight_tensor.data()[i] = 2.0f;
    }
    std::size_t weight_id = g.AddConstant(weight_tensor, "weight");
    runtime::Tensor bias_tensor({2});
    for (std::size_t i = 0; i < bias_tensor.numel(); ++i) {
        bias_tensor.data()[i] = 1.0f;
    }
    std::size_t bias_id = g.AddConstant(bias_tensor, "bias");
    std::size_t matmul_id = g.AddOp(runtime::OpType::kMatMul, {input_id, weight_id}, "matmul");
    std::size_t add_id = g.AddOp(runtime::OpType::kAdd, {matmul_id, bias_id}, "add");
    std::size_t relu_id = g.AddOp(runtime::OpType::kRelu, {add_id}, "relu");

    // First run, same as FullPipeline: input=[3,3] -> output=[13,13].
    runtime::Tensor input1({1, 2});
    for (std::size_t i = 0; i < input1.numel(); ++i) {
        input1.data()[i] = 3.0f;
    }
    runtime::Tensor output1 = g.Execute({{input_id, input1}}, relu_id);
    for (std::size_t i = 0; i < output1.numel(); ++i) {
        EXPECT_FLOAT_EQ(output1.data()[i], 13.0f);
    }

    // Second run on the SAME graph, different input: [1,1] -> each
    // output entry is 1*2 + 1*2 + 1 (bias) = 5. If a node's value from
    // the first Execute() leaked into this one instead of being
    // recomputed, this would incorrectly still show 13.
    runtime::Tensor input2({1, 2});
    for (std::size_t i = 0; i < input2.numel(); ++i) {
        input2.data()[i] = 1.0f;
    }
    runtime::Tensor output2 = g.Execute({{input_id, input2}}, relu_id);
    for (std::size_t i = 0; i < output2.numel(); ++i) {
        EXPECT_FLOAT_EQ(output2.data()[i], 5.0f);
    }
}
//
// TEST(GraphTest, SingleInputReturnsItsOwnValue) { ... }
