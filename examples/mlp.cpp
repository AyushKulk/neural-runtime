// End-to-end demo: a small 2-layer MLP built and executed entirely
// through this runtime's Tensor/ops/Graph stack. This is the Week 1
// milestone closer — everything it uses (Tensor, MatMul/Add/Relu, Graph)
// is already implemented and tested; this file just assembles those
// pieces into something recognizable as "a neural network runs here."
//
// Architecture (typical small MLP, 2 Linear+Relu layers):
//   Input {1, 3}                         one example, 3 input features
//     -> MatMul(W1 {3,4}) -> Add(b1 {4}) -> Relu     hidden layer, 4 units
//     -> MatMul(W2 {4,2}) -> Add(b2 {2}) -> Relu     output layer, 2 units
//   Output {1, 2}
//
// Per this milestone's decision, weights are hand-written directly below
// rather than loaded from a file — keep them simple (e.g. small fixed
// values you can still sanity-check by hand), the point here is
// demonstrating the runtime works end to end, not the specific numbers.

#include <iostream>

#include "runtime/graph.h"
#include "runtime/ops.h"

int main() {
    // TODO: build the graph.
    //   1. runtime::Graph g;
    runtime::Graph g;
    //   2. std::size_t input_id = g.AddInput("x");
    std::size_t input_id = g.AddInput("x");
    //   3. Construct W1 {3,4}, b1 {4}, W2 {4,2}, b2 {2} as Tensors, fill
    //      them with values of your choosing (loop over .numel(), same
    //      pattern you've used in every test file so far), and register
    //      each with g.AddConstant(...).
    
    runtime::Tensor W1({3, 4});
    for (std::size_t i = 0; i < W1.numel(); ++i) {
        W1.data()[i] = 1.0f;  // Fill with your chosen values
    }
    std::size_t W1_id = g.AddConstant(W1, "W1");
    
    runtime::Tensor b1({4});
    for (std::size_t i = 0; i < b1.numel(); ++i) {
        b1.data()[i] = 0.5f;  // Fill with your chosen values
    }
    std::size_t b1_id = g.AddConstant(b1, "b1");
    
    runtime::Tensor W2({4, 2});
    for (std::size_t i = 0; i < W2.numel(); ++i) {
        W2.data()[i] = 2.0f;  // Fill with your chosen values
    }
    std::size_t W2_id = g.AddConstant(W2, "W2");    
    
    runtime::Tensor b2({2});
    for (std::size_t i = 0; i < b2.numel(); ++i) {
        b2.data()[i] = 1.0f;  // Fill with your chosen values
    }
    std::size_t b2_id = g.AddConstant(b2, "b2");
    //   4. Chain the first layer: g.AddOp(kMatMul, {input_id, w1_id}),
    //      then g.AddOp(kAdd, {matmul1_id, b1_id}), then
    //      g.AddOp(kRelu, {add1_id}).
    std::size_t matmul1_id = g.AddOp(runtime::OpType::kMatMul, {input_id, W1_id}, "matmul1");
    std::size_t add1_id = g.AddOp(runtime::OpType::kAdd, {matmul1_id, b1_id}, "add1");
    std::size_t relu1_id = g.AddOp(runtime::OpType::kRelu, {add1_id}, "relu1");
    //   5. Chain the second layer the same way, feeding the first
    //      layer's Relu output in as MatMul's left-hand input this time.
    std::size_t matmul2_id = g.AddOp(runtime::OpType::kMatMul, {relu1_id, W2_id}, "matmul2");
    std::size_t add2_id = g.AddOp(runtime::OpType::kAdd, {matmul2_id, b2_id}, "add2");
    std::size_t relu2_id = g.AddOp(runtime::OpType::kRelu, {add2_id}, "relu2");
    //   6. relu2_id is the final output
    std::size_t final_output_id = relu2_id;
    
    //   6. Build an actual input Tensor {1,3} with some values.
    runtime::Tensor your_input_tensor({1, 3});
    for (std::size_t i = 0; i < your_input_tensor.numel(); ++i) {
        your_input_tensor.data()[i] = 1.0f;  // Fill with your chosen values
    }
    //   7. g.Execute({{input_id, your_input_tensor}}, final_output_id) and
    //      print the result (loop over .numel(), print output.data()[i]).
    //
    g.Execute({{input_id, your_input_tensor}}, final_output_id);
    // Once it runs, hand-compute the expected output the same way you
    // did for the MatMul/Graph tests, and check the printed values match
    // — this file has no assertions of its own, so that check is on you.

    printf("Output Tensor:\n");
    runtime::Tensor output_tensor = g.Execute({{input_id, your_input_tensor}}, final_output_id);
    for (std::size_t i = 0; i < output_tensor.numel(); ++i) {
        std::cout << output_tensor.data()[i] << " ";
    }
    std::cout << std::endl;
    return 0;
}
