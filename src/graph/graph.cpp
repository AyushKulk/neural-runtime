#include "runtime/graph.h"

#include <cassert>
#include <optional>

#include "runtime/ops.h"

namespace runtime {

// TODO: implement. Append a Node{OpType::kInput, {}, name, std::nullopt}
// to nodes_ and return its index (nodes_.size() - 1 before the push, or
// equivalently size() after - 1).
std::size_t Graph::AddInput(const std::string& name) {
    nodes_.push_back(Node{OpType::kInput, {}, name, {}});
    return (nodes_.size() - 1);
}

// TODO: implement. Same shape as AddInput, but op_type is kConstant and
// value is set to std::move(value) up front — a Constant's value is known
// at graph-build time, not filled in later by Execute().
std::size_t Graph::AddConstant(Tensor value, const std::string& name) {
    nodes_.push_back(Node{OpType::kConstant, {}, name, std::move(value)});
    return (nodes_.size() - 1);
}

// TODO: implement. Append a Node with the given op_type/input_ids/name
// and an unset value (compute nodes only get their value during
// Execute()). Consider asserting every id in input_ids is < nodes_.size()
// — a caller passing a not-yet-created node ID is a contract violation
// worth catching immediately rather than segfaulting later during
// execution.
std::size_t Graph::AddOp(OpType op_type, std::vector<std::size_t> input_ids,
                          const std::string& name) {
    (void)op_type;
    (void)input_ids;
    (void)name;
    nodes_.push_back(Node{op_type, input_ids, name, {}});
    return (nodes_.size() - 1);
}

namespace {

// Per-node DFS state. Three states, not two — see the design discussion:
// Done lets a node visited via two different paths (e.g. a Constant two
// different ops both depend on) get skipped the second time without
// reprocessing; InProgress is what lets a cycle be detected (revisiting a
// node that's still mid-traversal can only happen if the graph isn't
// actually a DAG).
enum class VisitState { kUnvisited, kInProgress, kDone };

// Post-order DFS from `id`: visits (and appends to `order`) every
// dependency of `id` before appending `id` itself, which is exactly what
// makes the final `order` a valid execution order.
void Visit(std::size_t id, const std::vector<Node>& nodes,
           std::vector<VisitState>& state, std::vector<std::size_t>& order) {
    //   1. If state[id] == kDone, return immediately — already fully
    //      processed via some other path, nothing more to do.
    if (state[id] == VisitState::kDone) {
        return;
    }
    //   2. assert(state[id] != kInProgress) — if this fires, you've
    //      found a cycle (Visit(id) called again while still inside an
    //      earlier, unfinished call to Visit(id)).
    assert(state[id] != VisitState::kInProgress);
    //   3. Set state[id] = kInProgress.
    state[id] = VisitState::kInProgress;
    //   4. For each dep in nodes[id].input_ids, call Visit(dep, nodes,
    //      state, order) — recurse before doing anything else with `id`.
    for (std::size_t dep : nodes[id].input_ids) {
        Visit(dep, nodes, state, order);
    }
    //   5. Set state[id] = kDone.
    state[id] = VisitState::kDone;
    //   6. order.push_back(id) — AFTER the recursive calls, not before;
    //      that ordering is what makes this post-order.
    order.push_back(id);
    (void)id;
    (void)nodes;
    (void)state;
    (void)order;
}

// Given the full node list and the ID you actually want (output_id),
// returns node IDs in an order where every node appears after everything
// in its input_ids. Only visits nodes reachable from output_id, not
// every node in `nodes` — a graph can have unused branches.
std::vector<std::size_t> TopologicalOrder(const std::vector<Node>& nodes,
                                           std::size_t output_id) {
    // TODO:
    //   1. Allocate `state`, sized nodes.size(), all kUnvisited.
    std::vector<VisitState> state(nodes.size(), VisitState::kUnvisited);
    //   2. Allocate `order`, empty, and reserve nodes.size() if you want
    //      to avoid reallocation (optional, not required for
    //      correctness).
    std::vector<std::size_t> order;
    order.reserve(nodes.size());
    //   3. Call Visit(output_id, nodes, state, order).
    Visit(output_id, nodes, state, order);
    //   4. Return order.
    (void)nodes;
    (void)output_id;
    return order;
}

}  // namespace

// TODO: implement. Roughly:
//   1. Get the execution order via TopologicalOrder(nodes_, output_id).
//   2. For each node ID in that order:
//      - kInput: look it up in `inputs` (by node ID) and set
//        nodes_[id].value.
//      - kConstant: value is already set from AddConstant, nothing to do.
//      - kMatMul/kAdd/kRelu: pull the already-computed value(s) out of
//        nodes_[input_ids[...]].value (they must be set already, since
//        the topological order guarantees dependencies ran first —
//        assert this rather than silently dereferencing an empty
//        optional), call the matching ops.h function, store the result
//        in nodes_[id].value.
//   3. Return nodes_[output_id].value (it must be set by now).
Tensor Graph::Execute(const std::unordered_map<std::size_t, Tensor>& inputs,
                       std::size_t output_id) {
    (void)inputs;
    (void)output_id;

    for (std::size_t id : TopologicalOrder(nodes_, output_id)) {
        Node& node = nodes_[id];
        switch (node.op_type) {
            case OpType::kInput: {
                auto it = inputs.find(id);
                assert(it != inputs.end());
                node.value = it->second;
                break;
            }
            case OpType::kConstant:
                // Nothing to do; value is already set.
                break;
            case OpType::kMatMul: {
                assert(node.input_ids.size() == 2);
                const Tensor& A = nodes_[node.input_ids[0]].value.value();
                const Tensor& B = nodes_[node.input_ids[1]].value.value();
                node.value = MatMul(A, B);
                break;
            }
            case OpType::kAdd: {
                assert(node.input_ids.size() == 2);
                const Tensor& A = nodes_[node.input_ids[0]].value.value();
                const Tensor& B = nodes_[node.input_ids[1]].value.value();
                node.value = Add(A, B);
                break;
            }
            case OpType::kRelu: {
                assert(node.input_ids.size() == 1);
                const Tensor& A = nodes_[node.input_ids[0]].value.value();
                node.value = Relu(A);
                break;
            }
        }
    }

    return nodes_[output_id].value.value();
}

}  // namespace runtime
