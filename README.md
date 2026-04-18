# gitc0ffee

Brute-force vanity prefixes for git commit hashes.

Make your commits start with `c0ffee`, `deadbeef`, `badc0de` - or whatever makes you happy.

## Requirements

### macOS (Metal GPU)

- Apple Silicon (M1/M2/M3/M4)
- OpenSSL 3 (`brew install openssl@3` or build from source)
- Xcode Command Line Tools (`xcode-select --install`)

### Linux (CPU with AVX-512, SHA-NI, or AVX2)

- x86-64 CPU with AVX-512 (Intel Skylake-X+, AMD Zen 4+), SHA-NI (Intel Goldmont+, AMD Zen+), or AVX2 (Intel Haswell+, AMD Excavator+)
- OpenSSL development headers (`libssl-dev` / `openssl-devel`)
- GCC 10+ with C++20 support

## Install

From source:

```bash
make
make install          # installs to /usr/local/bin
```

To uninstall:

```bash
make uninstall
```

## Usage

```bash
git commit -am 'my changes'

# find a vanity hash (prints the new hash to stdout)
gitc0ffee c0ffee

# find it and update HEAD in one shot
gitc0ffee c0ffee --write
```

Odd-length prefixes work too: `bad`, `dec0de5`.

### Example output (macOS - Metal GPU)

```
Target Prefix  c0ffee (6 nibbles)
Padding        48 bytes (1 SHA1 block)
Object Size    305 bytes
Salt Offset    256

Device         Apple M4 Pro
Dispatch       128M x 3 in-flight

Found          c0ffeeb6f8bb842914960286cbd62490fd91b0af
Time           0.04s
Throughput     8.95 GH/s
Hashes Tried   0.40G
```

### Example output (Linux - CPU AVX-512)

```
Target Prefix  deadbeef (8 nibbles)
Padding        19 bytes (1 SHA1 block)
Object Size    881 bytes
Salt Offset    832

Device         CPU (192 threads, AVX-512, 16-way)

Found          deadbeef3f2c6bd3373376c7d821241260274513
Time           0.52s
Throughput     21.30 GH/s
Hashes Tried   11.27G
```

## Performance

### macOS (M4 Pro)

| Prefix | Nibbles | Expected hashes | Time |
|--------|---------|-----------------|------|
| `c0ffee` | 6 | ~16M | <0.1s |
| `c0ffeec0` | 8 | ~4.3B | ~2s |
| `c0ffeec0ff` | 10 | ~1.1T | ~5 min |

Sustained throughput: ~3.6 GH/s.

### Linux (AMD EPYC 9R14, 192 threads)

| Prefix | Nibbles | Expected hashes | Time |
|--------|---------|-----------------|------|
| `c0ffee` | 6 | ~16M | <0.1s |
| `c0ffeec0` | 8 | ~4.3B | ~0.2s |
| `c0ffeec0ff` | 10 | ~1.1T | ~1 min |

Sustained throughput: ~22 GH/s with AVX-512 (16-way), ~14 GH/s with SHA-NI (4-way), AVX2 (8-way) as universal fallback. Backend auto-selected at runtime.

## How it works

1. Read the current HEAD commit.
2. Append invisible trailing whitespace to the commit message: padding spaces for SHA1 block alignment, followed by a 48-bit salt encoded as spaces (0) and tabs (1).
3. Pre-compute SHA1 state for all blocks before the salt on the CPU.
4. On macOS: dispatch millions of GPU threads via Metal compute shaders. On Linux: dispatch across all CPU cores using AVX-512 (16-way SIMD), SHA-NI (4-way), or AVX2 (8-way), auto-selected at runtime.
5. Each thread/core tries a different salt, runs 80 SHA1 rounds, and checks the prefix.
6. Write the winning commit object to the git store.
7. Optionally update HEAD.

The salt and padding are completely invisible - they're trailing whitespace that doesn't show up in `git log`, `git show`, GitHub, or any standard git UI. Your commit message stays clean.

### Key optimizations

- **Invisible salt** - 48-bit salt encoded as trailing spaces and tabs. No visible headers or markers.
- **Always single-block** - salt at the end of the object means the solver always processes exactly 1 SHA1 block (80 rounds), regardless of commit message length.
- **Pre-computed SHA1 state** - CPU hashes all blocks before the salt block once.
- **Nibble-level prefix check** - compares SHA1 words with 4-bit mask granularity. Supports odd-length prefixes.

#### macOS (Metal GPU)

- **Fully unrolled shader** - scalar variables with fused message schedule expansion. No arrays, no loops, zero register spill.
- **Triple-buffered dispatch** - three Metal command buffers in flight for zero GPU idle time.

#### Linux (CPU)

- **Runtime backend selection** - auto-detects AVX-512 > SHA-NI > AVX2 at startup via CPUID, picks the fastest available path.
- **AVX-512 16-way SIMD** - processes 16 independent SHA1 hashes in parallel using 512-bit registers. Uses `vpternlogd` for 1-uop round functions, `vprold` for native rotate, and `vpermd` for SIMD salt LUT lookup.
- **SHA-NI 4-way interleaved** - four independent SHA1 streams saturate the `sha1rnds4` pipeline (5-cycle latency, 1-cycle throughput). Deferred E computation skips finalization on non-matches.
- **AVX2 8-way SIMD** - processes 8 independent SHA1 hashes using 256-bit registers with `vgatherdps` for salt LUT lookup. Universal x86-64 fallback for CPUs without AVX-512 or SHA-NI.
- **Nibble LUT salt encoding** - builds SHA1 message words directly from the 48-bit salt via lookup table, skipping byte-level encoding and byte-swap shuffles.
- **Incremental salt update** - only recomputes the 4 low words each iteration; the 8 high words are reused across 65536 consecutive salts.
- **Autonomous workers** - each thread owns a contiguous salt range with no synchronization barriers. Scales from 4 cores to 192+.

## Testing

```bash
make test
```

Runs commit template tests, solver correctness tests, and end-to-end integration tests (TAP format).

## Project structure

```
src/
  main.cpp            CLI entry point (auto-selects GPU on macOS, CPU on Linux)
  types.h             Shared types (HexDigest, ObjectTemplate, SolveResult)
  solver.h            Solver interface
  git/
    commit.cpp/.h     Commit parsing and template construction
    git.cpp/.h        Git plumbing (rev-parse, cat-file, hash-object)
  cpu/
    solver.cpp        Multi-threaded dispatch and SHA1 pre-computation (Linux)
    common.h          Shared types and constants for CPU backends
    avx512.cpp        AVX-512 backend: 16-way SIMD SHA1
    sha_ni.cpp        SHA-NI backend: 4-way interleaved SHA1
    avx2.cpp          AVX2 backend: 8-way SIMD SHA1
  gpu/
    metal_solver.mm   Metal GPU dispatch and SHA1 pre-computation (macOS)
    sha1.metal        Metal compute shader (SHA1 brute-force kernel)
tests/
  test_commit.cpp     Template alignment and SHA1 consistency tests
  test_cpu.cpp        CPU solver correctness tests
  test_gpu.mm         GPU solver correctness tests (macOS)
  test_e2e.sh         End-to-end integration tests
```

## Prefix ideas

Any hex string works. For inspiration see [Hexspeak](https://en.wikipedia.org/wiki/Hexspeak):

`c0ffee` * `deadbeef` * `badc0de` * `cafebabe` * `f00d` * `beef` * `face` * `decade` * `bad` * `dec0de5`

## License

MIT - see [LICENSE](LICENSE).
