# gitc0ffee

Brute-force vanity prefixes for git commit hashes using Metal GPU compute.

Make your commits start with `c0ffee`, `deadbeef`, `badc0de` — or whatever makes you happy.

## Requirements

- macOS with Apple Silicon (M1/M2/M3/M4)
- OpenSSL 3 (`brew install openssl@3` or build from source)
- Xcode Command Line Tools (`xcode-select --install`)

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
gitc0ffee --prefix c0ffee

# find it and update HEAD in one shot
gitc0ffee --prefix c0ffee --update-ref
```

Odd-length prefixes work too: `--prefix bad`, `--prefix dec0de5`.

### Example output

```
Target Prefix  c0ffee (6 nibbles)
Padding        48 bytes (1 SHA1 block)
Object Size    305 bytes
Salt Offset    256

Device         Apple M4 Pro
Dispatch       128M × 3 in-flight

✓ Found        c0ffeeb6f8bb842914960286cbd62490fd91b0af
Time           0.04s
Throughput     8.95 GH/s
Hashes Tried   0.40G
```

## Performance

Measured on a MacBook Pro 16" 2024 (M4 Pro):

| Prefix | Nibbles | Expected hashes | Time |
|--------|---------|-----------------|------|
| `c0ffee` | 6 | ~16M | <0.1s |
| `c0ffeec0` | 8 | ~4.3B | ~2s |
| `c0ffeec0ff` | 10 | ~1.1T | ~5 min |

Sustained throughput: ~3.6 GH/s.

## How it works

1. Read the current HEAD commit.
2. Append invisible trailing whitespace to the commit message: padding spaces for SHA1 block alignment, followed by a 48-bit salt encoded as spaces (0) and tabs (1).
3. Pre-compute SHA1 state for all blocks before the salt on the CPU.
4. Dispatch millions of GPU threads via Metal compute shaders. Each thread tries a different salt, runs 80 SHA1 rounds, and checks the prefix.
5. Write the winning commit object to the git store.
6. Optionally update HEAD.

The salt and padding are completely invisible — they're trailing whitespace that doesn't show up in `git log`, `git show`, GitHub, or any standard git UI. Your commit message stays clean.

### Key optimizations

- **Invisible salt** — 48-bit salt encoded as trailing spaces and tabs via a LUT. No visible headers or markers.
- **Always single-block** — salt at the end of the object means the GPU always processes exactly 1 SHA1 block (80 rounds), regardless of commit message length.
- **Fully unrolled shader** — scalar variables with fused message schedule expansion. No arrays, no loops, zero register spill.
- **Pre-computed SHA1 state** — CPU hashes all blocks before the salt block.
- **Triple-buffered dispatch** — three Metal command buffers in flight for zero GPU idle time.
- **Nibble-level prefix check** — compares SHA1 words with 4-bit mask granularity. Supports odd-length prefixes.

## Testing

```bash
make test
```

Runs 15 tests (TAP format) covering short/long/unicode authors, merges, GPG signatures, multi-line messages, extra headers, and more.

## Project structure

```
src/
  main.cpp          CLI entry point
  commit.cpp/.h     Commit parsing and template construction
  git.cpp/.h        Git plumbing (rev-parse, cat-file, hash-object)
  gpu_solver.mm/.h  Metal GPU dispatch and SHA1 pre-computation
  sha1.metal        Metal compute shader (SHA1 brute-force kernel)
  types.h           Shared types (HexDigest, ObjectTemplate, SolveResult)
tests/
  test_commit.cpp   Template alignment and SHA1 consistency tests
```

## Prefix ideas

Any hex string works. For inspiration see [Hexspeak](https://en.wikipedia.org/wiki/Hexspeak):

`c0ffee` · `deadbeef` · `badc0de` · `cafebabe` · `f00d` · `beef` · `face` · `decade` · `bad` · `dec0de5`

## License

MIT — see [LICENSE](LICENSE).
