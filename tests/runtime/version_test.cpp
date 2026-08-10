#include "runtime/version.h"

#include <gtest/gtest.h>

// Scaffolding-milestone test: exercises the build/link/test pipeline
// end to end (GoogleTest + CTest + runtime_core) before any real tensor
// or graph code exists.
TEST(VersionTest, ReturnsExpectedTriple) {
    const runtime::Version v = runtime::GetVersion();
    EXPECT_EQ(v.major, 0);
    EXPECT_EQ(v.minor, 1);
    EXPECT_EQ(v.patch, 0);
}

TEST(VersionTest, StringMatchesTriple) {
    EXPECT_STREQ(runtime::GetVersionString(), "0.1.0");
}
