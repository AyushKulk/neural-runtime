#include "runtime/version.h"

namespace runtime {

Version GetVersion() { return Version{0, 1, 0}; }

const char* GetVersionString() { return "0.1.0"; }

}  // namespace runtime
