#pragma once

namespace runtime {

// Simple semantic version triple for the runtime library itself. Bumped by
// hand as milestones land; not tied to any model format's versioning.
struct Version {
    int major;
    int minor;
    int patch;
};

// Returns the current library version. Exists (rather than a constant) so
// unit tests have something real to link against and assert on, exercising
// the build/test wiring before any tensor/graph code exists.
Version GetVersion();

// Human-readable "MAJOR.MINOR.PATCH" string for the version above.
const char* GetVersionString();

}  // namespace runtime
