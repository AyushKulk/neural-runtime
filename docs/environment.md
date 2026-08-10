# Development Environment

Verified working setup as of the scaffolding milestone (2026-08-10).

## Machine

- macOS 26.4.1 (Darwin 25.4.0), arm64 (Apple Silicon)

## Toolchain

| Tool          | Version                              | Source                          |
|---------------|---------------------------------------|----------------------------------|
| Compiler      | Apple clang 17.0.0 (clang-1700.0.13.5) | Xcode Command Line Tools (preinstalled) |
| CMake         | 4.4.2                                  | Homebrew (`brew install cmake`) |
| Ninja         | 1.13.2                                 | Homebrew (`brew install ninja`) |
| GNU Make      | 3.81                                   | Xcode Command Line Tools (preinstalled) |
| Git           | 2.53.0                                 | Preinstalled                    |
| Python        | 3.14.0                                 | Preinstalled                    |

CMake and Ninja were **not** preinstalled and had to be added via Homebrew.
Apple clang was already present and is used as both the C and C++ compiler;
it supports C++20 (`-std=c++20`), which is all this project currently
requires.

Ninja is preferred for local builds (faster incremental rebuilds, cleaner
parallel output), but the project is generator-agnostic: it was verified to
configure and build identically with the default Unix Makefiles generator,
so a machine without Ninja can still build with plain `cmake --build`.

## GoogleTest

Rather than depending on a system-installed GoogleTest (which varies in
version/availability across macOS and Linux), the test target pulls
GoogleTest v1.15.2 via CMake's `FetchContent` at configure time. This keeps
`cmake -S . -B build` + `cmake --build build` sufficient to get a working
test binary on a fresh machine, at the cost of a one-time source download
into `build/_deps/`.

## Verified build paths

Both of the following were run from a clean checkout and produced a
successful build + all tests passing:

```bash
# Ninja (preferred)
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure

# Default generator (Unix Makefiles) — sanity check for portability
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc || sysctl -n hw.ncpu)"
ctest --test-dir build --output-on-failure
```

## Known non-issues

- `FetchContent` emits a `CMP0135` author warning about
  `DOWNLOAD_EXTRACT_TIMESTAMP` on first configure; this is silenced by
  explicitly setting `DOWNLOAD_EXTRACT_TIMESTAMP TRUE` in
  `tests/CMakeLists.txt`.

## Linux

Not yet verified on Linux in this environment. Expected to work unmodified
with a C++20-capable GCC (>= 10) or Clang (>= 12), CMake >= 3.20, and either
Ninja or Make — nothing in the current CMake configuration is
platform-specific. This will be confirmed once CI is set up (planned for
Week 3).
