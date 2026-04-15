// cpu_solver.cpp - multi-threaded SHA1 brute-force with runtime dispatch
//
// Three backends (selected by CPUID, priority order):
//   1. AVX-512:  16-way SIMD SHA1 using 512-bit registers
//   2. SHA-NI:   4-way interleaved using x86 SHA extensions
//   3. AVX2:     8-way SIMD SHA1 using 256-bit registers (universal fallback)

#ifndef __x86_64__
#error "CPU solver requires x86_64 (AVX2/AVX-512/SHA-NI)"
#endif

#include "solver.h"
#include "cpu/common.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <openssl/sha.h>
#include <thread>
#include <vector>
#ifdef __linux__
#include <sched.h>
#include <cpuid.h>
#endif

// ---------------------------------------------------------------------------
// CPU feature detection
// ---------------------------------------------------------------------------

static bool has_avx512() {
#if defined(__x86_64__) && defined(__linux__)
    unsigned eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
        return ((ebx >> 16) & 1) && ((ebx >> 30) & 1);  // AVX512F + AVX512BW
#endif
    return false;
}

static bool has_sha_ni() {
#if defined(__x86_64__) && defined(__linux__)
    unsigned eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx))
        return (ebx >> 29) & 1;
#endif
    return false;
}

// ---------------------------------------------------------------------------
// Prefix → mask/target conversion
// ---------------------------------------------------------------------------

static void build_prefix_masks(const std::string& prefix_hex, CPUParams& p) {
    p.prefix_len = static_cast<uint32_t>(prefix_hex.size());

    auto bytes = hex_string_to_bytes(prefix_hex);
    uint32_t words[5] = {};
    for (size_t i = 0; i < bytes.size(); ++i)
        words[i / 4] |= uint32_t(bytes[i]) << ((3 - (i % 4)) * 8);

    uint32_t n = p.prefix_len;
    p.mask0   = (n >= 8) ? ~0u : (~0u << ((8 - n) * 4));
    p.target0 = words[0] & p.mask0;

    if (n > 8) {
        uint32_t rem = n - 8;
        p.mask1   = (rem >= 8) ? ~0u : (~0u << ((8 - rem) * 4));
        p.target1 = words[1] & p.mask1;
    }
}

// ---------------------------------------------------------------------------
// SHA1 pre-state + final block
//
// The git object is laid out as:
//   [header blocks...][salt (48 bytes) | tail | 0x80 | padding | length]
//                      ^--- salt_offset (64-byte aligned)
//
// We SHA1-compress everything before the salt block, then hand the final
// block (with salt words zeroed) to the SIMD workers as msg_words[0..15].
// ---------------------------------------------------------------------------

static void build_sha1_state(const ObjectTemplate& tpl, CPUParams& p) {
    uint32_t total = static_cast<uint32_t>(tpl.bytes.size());
    int pre_bytes  = (tpl.salt_offset / 64) * 64;  // always == salt_offset

    // Compress all full blocks before the salt
    SHA_CTX ctx;
    SHA1_Init(&ctx);
    if (pre_bytes > 0)
        SHA1_Update(&ctx, tpl.bytes.data(), pre_bytes);

    p.pre_state[0] = ctx.h0;  p.pre_state[1] = ctx.h1;
    p.pre_state[2] = ctx.h2;  p.pre_state[3] = ctx.h3;
    p.pre_state[4] = ctx.h4;

    // Build the final 64-byte block with zeroed salt + SHA1 padding
    uint8_t block[64] = {};
    uint32_t tail_len = total - pre_bytes;
    memcpy(block, tpl.bytes.data() + pre_bytes, tail_len);
    memset(block, 0, kSaltBytes);           // zero the salt region
    block[tail_len] = 0x80;                 // SHA1 terminator

    uint64_t bit_len = uint64_t(total) * 8; // big-endian length
    for (int i = 0; i < 8; ++i)
        block[63 - i] = uint8_t(bit_len >> (i * 8));

    // Convert to 16 big-endian 32-bit words for the SIMD workers
    for (int i = 0; i < 16; ++i) {
        int o = i * 4;
        p.msg_words[i] = (uint32_t(block[o])   << 24) | (uint32_t(block[o+1]) << 16) |
                          (uint32_t(block[o+2]) <<  8) |  uint32_t(block[o+3]);
    }

    // SHA-NI needs words 12-15 as an __m128i
    p.msg3_const = _mm_set_epi32((int)p.msg_words[12], (int)p.msg_words[13],
                                 (int)p.msg_words[14], (int)p.msg_words[15]);
}

// ---------------------------------------------------------------------------
// Backend selection
// ---------------------------------------------------------------------------

struct Backend {
    WorkerFn    fn;
    const char* name;
};

static std::optional<Backend> select_backend(const std::string& override) {
    if (override.empty()) {
        if (has_avx512()) return Backend{avx512_worker, "AVX-512, 16-way"};
        if (has_sha_ni()) return Backend{sha_ni_worker,  "SHA-NI, 4-way"};
        return Backend{avx2_worker, "AVX2, 8-way"};
    }

    if (override == "avx512") {
        if (!has_avx512()) { fprintf(stderr, "Error: AVX-512 not supported\n"); return std::nullopt; }
        return Backend{avx512_worker, "AVX-512, 16-way"};
    }
    if (override == "sha-ni") {
        if (!has_sha_ni()) { fprintf(stderr, "Error: SHA-NI not supported\n"); return std::nullopt; }
        return Backend{sha_ni_worker, "SHA-NI, 4-way"};
    }
    if (override == "avx2")
        return Backend{avx2_worker, "AVX2, 8-way"};

    fprintf(stderr, "Error: unknown backend '%s'\n", override.c_str());
    return std::nullopt;
}

// ---------------------------------------------------------------------------
// Thread count
// ---------------------------------------------------------------------------

static unsigned pick_thread_count() {
    unsigned n = std::thread::hardware_concurrency();
    if (n == 0) n = 4;

#ifdef __linux__
    cpu_set_t cpuset;
    if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0) {
        unsigned avail = static_cast<unsigned>(CPU_COUNT(&cpuset));
        if (avail > 0 && avail < n) n = avail;
    }
#endif
    return n;
}

// ---------------------------------------------------------------------------
// Progress monitor  (runs on the main thread while workers are busy)
// ---------------------------------------------------------------------------

static void print_progress(const std::vector<WorkerResult>& results,
                           const std::atomic<bool>& done,
                           std::chrono::steady_clock::time_point t0) {
    int next_sec = 1;
    while (!done.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        int secs = int(std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count());
        if (secs < next_sec) continue;
        next_sec = secs + 1;

        uint64_t total = 0;
        for (auto& r : results) total += r.hashes;

        fprintf(stderr, "%.2fG hashes | %.2f GH/s | %ds elapsed\n",
                double(total) / 1e9, double(total) / secs / 1e9, secs);
    }
}

// ---------------------------------------------------------------------------
// Result extraction
// ---------------------------------------------------------------------------

static HexDigest hash_words_to_hex(const uint32_t h[5]) {
    static constexpr char hx[] = "0123456789abcdef";
    HexDigest out;
    for (int i = 0; i < 5; ++i) {
        uint32_t w = h[i];
        out[i*8+0] = hx[(w >> 28) & 0xF];  out[i*8+1] = hx[(w >> 24) & 0xF];
        out[i*8+2] = hx[(w >> 20) & 0xF];  out[i*8+3] = hx[(w >> 16) & 0xF];
        out[i*8+4] = hx[(w >> 12) & 0xF];  out[i*8+5] = hx[(w >>  8) & 0xF];
        out[i*8+6] = hx[(w >>  4) & 0xF];  out[i*8+7] = hx[ w        & 0xF];
    }
    return out;
}

static int find_winner(const std::vector<WorkerResult>& results) {
    int best = -1;
    for (int t = 0; t < (int)results.size(); ++t) {
        if (!results[t].found) continue;
        if (best < 0 || results[t].salt < results[best].salt)
            best = t;
    }
    return best;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

// SIMD batch width — thread salt ranges are aligned to this so that
// nibble-carry across lanes never happens within a single SIMD batch.
static constexpr uint64_t kSimdAlign = 16;

// 48-bit salt space
static constexpr uint64_t kMaxSalt = (1ULL << 48) - 1;

std::optional<SolveResult> solve(const ObjectTemplate& tpl,
                                 const std::string& prefix_hex,
                                 const std::string& backend_override) {
    // --- Prepare --------------------------------------------------------
    CPUParams params{};
    build_prefix_masks(prefix_hex, params);
    build_sha1_state(tpl, params);

    auto be = select_backend(backend_override);
    if (!be) return std::nullopt;

    unsigned nthreads = pick_thread_count();
    fprintf(stderr, "Device         CPU (%u threads, %s)\n\n", nthreads, be->name);

    // --- Launch workers -------------------------------------------------
    std::atomic<bool> global_found{false};
    std::vector<WorkerResult> results(nthreads);
    std::vector<std::thread>  threads(nthreads);

    auto t0 = std::chrono::steady_clock::now();

    uint64_t per_thread = kMaxSalt / nthreads;
    for (unsigned t = 0; t < nthreads; ++t) {
        // Round each boundary up to kSimdAlign so SIMD batches never
        // straddle a nibble-carry boundary (see commit 0e98db3).
        uint64_t start = (uint64_t(t)     * per_thread + (kSimdAlign - 1)) & ~(kSimdAlign - 1);
        uint64_t end   = (uint64_t(t + 1) * per_thread + (kSimdAlign - 1)) & ~(kSimdAlign - 1);
        if (t == 0)              start = 0;
        if (t == nthreads - 1)   end   = kMaxSalt;

        threads[t] = std::thread(be->fn, std::cref(params), start, end,
                                 std::ref(global_found), std::ref(results[t]));
    }

    // --- Wait -----------------------------------------------------------
    print_progress(results, global_found, t0);
    for (auto& th : threads) th.join();

    // --- Collect --------------------------------------------------------
    uint64_t total_hashes = 0;
    for (auto& r : results) total_hashes += r.hashes;

    int winner = find_winner(results);
    if (winner < 0) return std::nullopt;

    // Build the final payload with the winning salt
    auto out = tpl;
    out.set_salt(results[winner].salt);

    SolveResult sr;
    sr.payload = out.payload();
    sr.hash    = hash_words_to_hex(results[winner].hash);

    double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr,
        "Found          %s\n"
        "Time           %.2fs\n"
        "Throughput     %.2f GH/s\n"
        "Hashes Tried   %.2fG\n",
        hex_digest_to_string(sr.hash).c_str(),
        secs, double(total_hashes) / secs / 1e9, double(total_hashes) / 1e9);

    return sr;
}
