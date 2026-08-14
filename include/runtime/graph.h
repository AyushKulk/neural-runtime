#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "runtime/tensor.h"

namespace runtime {

// What kind of node this is. Input/Constant are not computed — they hold
// or receive a Tensor directly. The rest dispatch through the executor's
// switch statement to the matching free function in ops.h.
enum class OpType {
    kInput,     // Placeholder — value supplied per Execute() call.
    kConstant,  // Fixed data baked into the graph at construction time
                // (e.g. weights/bias — what a MatMul+Add layer's bias
                // ends up as).
    kMatMul,
    kAdd,
    kRelu,
};

// A single graph node. Which fields are meaningful depends on op_type:
//   - kInput:    input_ids empty, value unset until Execute() fills it in.
//   - kConstant: input_ids empty, value set once at construction (AddConstant).
//   - otherwise: input_ids are node IDs (indices into Graph's node list)
//     for this op's operands, in the order the op expects them (e.g. for
//     kMatMul, input_ids[0] is A, input_ids[1] is B). value is unset
//     until Execute() computes it.
struct Node {
    OpType op_type;
    std::vector<std::size_t> input_ids;
    std::string name;  // Optional; useful for debugging and, later, for
                        // matching against ONNX node/tensor names.
    std::optional<Tensor> value;
};

// Graph owns all nodes contiguously (index-based, not pointer-based — see
// the design discussion this milestone started from). A node ID is just
// its index into nodes_; AddInput/AddConstant/AddOp all return the ID of
// the node they created, which is what you pass as an input_id when
// wiring up a later node, and what you pass to Execute() to say which
// node's value you want out.
class Graph {
public:
    // Adds a placeholder node. Its value must be provided via the
    // `inputs` map passed to Execute() every call.
    std::size_t AddInput(const std::string& name);

    // Adds a node with a fixed Tensor baked in at graph-construction time
    // (e.g. a weight or bias loaded once, not supplied per-call).
    std::size_t AddConstant(Tensor value, const std::string& name = "");

    // Adds a compute node. `input_ids` must reference nodes that already
    // exist in this graph (i.e. graphs are built in dependency order —
    // you can't wire a node to an input that doesn't exist yet). Does
    // NOT execute anything; this only records graph structure.
    std::size_t AddOp(OpType op_type, std::vector<std::size_t> input_ids,
                       const std::string& name = "");

    // Runs the graph: fills in each Input node's value from `inputs`
    // (keyed by the node ID AddInput returned), then executes every node
    // needed to produce `output_id` in dependency order, and returns that
    // node's computed value.
    //
    // TODO (design decision to make while implementing, not before):
    // should this assert every kInput node has an entry in `inputs`, or
    // only the ones actually reachable from output_id? Reachable-only is
    // more permissive (a graph can have unused inputs) but means a typo'd
    // key silently does nothing rather than failing loudly. Pick one
    // deliberately.
    Tensor Execute(const std::unordered_map<std::size_t, Tensor>& inputs,
                   std::size_t output_id);

private:
    // TODO: topological sort. The graph's structure doesn't change
    // between Execute() calls (only Input values do), so this should be
    // computed once and reused — not re-derived on every call. Decide:
    // compute it lazily on first Execute() and cache it, or require an
    // explicit Graph::Finalize() step? Either is reasonable; lazy-cache
    // is less ceremony, explicit Finalize() is more honest about there
    // being a "done building, ready to run" state transition.
    //
    // Implementation-wise: DFS from output_id, visiting a node's
    // input_ids before appending the node itself to the order (postorder
    // = valid execution order). Worth tracking "currently visiting" vs
    // "done" per node so a cycle (which should never happen in a
    // correctly-built graph, but especially once ONNX loading exists,
    // malformed input is possible) gets caught with an assert rather than
    // infinite-looping.

    std::vector<Node> nodes_;
};

}  // namespace runtime
