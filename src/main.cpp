#include <iostream>

#include "runtime/version.h"

// Placeholder CLI entry point. This exists at this stage only to prove the
// toolchain (compiler, CMake, linking against the runtime_core library)
// works end to end. Real subcommands (e.g. `runtime run model.onnx`) land
// in later milestones once the tensor/graph/execution layers exist.
int main() {
    std::cout << "neural-runtime v" << runtime::GetVersionString() << "\n";
    std::cout << "Scaffolding milestone: environment and build verified.\n";
    return 0;
}
