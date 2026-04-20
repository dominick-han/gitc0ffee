// solver.cpp - multi-threaded SHA-1 brute-force with runtime dispatch.
//
// Three backends, selected by CPUID in priority order:
//   1. AVX-512: 16-way SIMD using 512-bit registers
//   2. SHA-NI:   4-way interleaved x86 SHA extensions
//   3. AVX2:     8-way SIMD using 256-bit registers (universal fallback)

#ifndef __x86_64__
#error "CPU solver requires x86_64 (AVX2/AVX-512/SHA-NI)"
#endif

#include "solver.h"
#include "cpu/common.h"

#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <openssl/sha.h>
#include <thread>
#include <vector>
#ifdef __linux__
#include <cpuid.h>
#include <sched.h>
#endif

// --- CPU feature detection -------------------------------------------------

// Test bit `b` of CPUID leaf 7 subleaf 0, register ebx.
static bool cpuid7(unsigned b) {
#ifdef __linux__
    unsigned a, x, c, d;
    return __get_cpuid_count(7, 0, &a, &x, &c, &d) && ((x >> b) & 1u);
#else
    return false;
#endif
}
static bool has_avx512() { return cpuid7(16) && cpuid7(30); }  // F + BW
static bool has_sha_ni() { return cpuid7(29); }

// --- Prefix -> mask/target -------------------------------------------------
//
// Pack up to the first 16 hex nibbles of `prefix` into two 32-bit words
// plus matching masks. Trailing words/masks stay zero, correctly interpreted
// by the backends as "don't care".

static void build_prefix_masks(const std::string& prefix, CPUParams& p) {
    static constexpr auto kNibble = [] {
        std::array<uint8_t, 256> t{};
        for (int c = '0'; c <= '9'; ++c) t[c] = c - '0';
        for (int c = 'a'; c <= 'f'; ++c) t[c] = c - 'a' + 10;
        for (int c = 'A'; c <= 'F'; ++c) t[c] = c - 'A' + 10;
        return t;
    }();

    p.prefix_len = (uint32_t)prefix.size();
    uint32_t words[2] = {}, masks[2] = {};
    for (size_t i = 0; i < prefix.size() && i < 16; ++i) {
        const int slot  = int(i) / 8;
        const int shift = (7 - (int(i) & 7)) * 4;
        words[slot] |= uint32_t(kNibble[(uint8_t)prefix[i]]) << shift;
        masks[slot] |= 0xFu << shift;
    }
    p.target0 = words[0]; p.mask0 = masks[0];
    p.target1 = words[1]; p.mask1 = masks[1];
}

// --- SHA-1 pre-state + final block ----------------------------------------
//
// Git object layout:
//   [header blocks...][salt (48 B) | tail | 0x80 | padding | length]
//                      ^--- salt_offset (64-byte aligned)
//
// Compress everything before the salt once; hand the final block (salt
// region zeroed) to the SIMD workers as msg_words[0..15].

static void build_sha1_state(const ObjectTemplate& tpl, CPUParams& p) {
    const uint32_t total = (uint32_t)tpl.bytes.size();
    const int      pre   = tpl.salt_offset;   // 64-byte aligned by prepare_template

    SHA_CTX ctx;
    SHA1_Init(&ctx);
    if (pre > 0) SHA1_Update(&ctx, tpl.bytes.data(), pre);
    for (int i = 0; i < 5; ++i) p.pre_state[i] = (&ctx.h0)[i];

    // Final block = zeroed salt region + tail bytes + SHA-1 padding.
    uint8_t block[64] = {};
    const uint32_t tail_len = total - pre;
    std::memcpy(block + kSaltBytes, tpl.bytes.data() + pre + kSaltBytes,
                tail_len - kSaltBytes);
    block[tail_len] = 0x80;
    const uint64_t bit_len = uint64_t(total) * 8;
    for (int i = 0; i < 8; ++i) block[63 - i] = uint8_t(bit_len >> (i * 8));

    for (int i = 0; i < 16; ++i) {
        uint32_t w;
        std::memcpy(&w, block + i * 4, 4);
        p.msg_words[i] = __builtin_bswap32(w);
    }
    p.msg3_const = _mm_set_epi32((int)p.msg_words[12], (int)p.msg_words[13],
                                 (int)p.msg_words[14], (int)p.msg_words[15]);
}

// --- Backend selection -----------------------------------------------------

struct Backend {
    const char* flag;
    const char* name;
    WorkerFn    fn;
    bool      (*supported)();   // nullptr = always supported
};

static constexpr Backend kBackends[] = {
    {"avx512", "AVX-512, 16-way", avx512_worker, has_avx512},
    {"sha-ni", "SHA-NI, 4-way",   sha_ni_worker, has_sha_ni},
    {"avx2",   "AVX2, 8-way",     avx2_worker,   nullptr},
};

static const Backend* select_backend(const std::string& want) {
    for (auto& b : kBackends) {
        const bool ok = !b.supported || b.supported();
        if (want.empty()) { if (ok) return &b; continue; }
        if (want != b.flag) continue;
        if (!ok) { fprintf(stderr, "Error: %s not supported\n", b.name); return nullptr; }
        return &b;
    }
    if (!want.empty()) fprintf(stderr, "Error: unknown backend '%s'\n", want.c_str());
    return nullptr;
}

// --- Thread count ----------------------------------------------------------

static unsigned pick_thread_count() {
    unsigned n = std::thread::hardware_concurrency();
#ifdef __linux__
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0)
        if (unsigned avail = (unsigned)CPU_COUNT(&set); avail > 0) n = std::min(n, avail);
#endif
    return n > 0 ? n : 4;
}

// --- Progress monitor ------------------------------------------------------

static void print_progress(const std::vector<WorkerResult>& results,
                           const std::atomic<bool>& done,
                           std::chrono::steady_clock::time_point t0) {
    int next_sec = 1;
    while (!done.load(std::memory_order_relaxed)) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const int secs = int(std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t0).count());
        if (secs < next_sec) continue;
        next_sec = secs + 1;
        uint64_t total = 0;
        for (auto& r : results) total += r.hashes;
        fprintf(stderr, "%.2fG hashes | %.2f GH/s | %ds elapsed\n",
                double(total) / 1e9, double(total) / secs / 1e9, secs);
    }
}

// --- Public API ------------------------------------------------------------

// SIMD batch width — thread salt ranges are aligned to this so a SIMD batch
// never straddles a nibble-carry boundary.
static constexpr uint64_t kSimdAlign = 16;
// 48-bit salt space.
static constexpr uint64_t kMaxSalt = (1ULL << 48) - 1;

std::optional<SolveResult> solve(const ObjectTemplate& tpl,
                                 const std::string& prefix_hex,
                                 const std::string& backend_override) {
    CPUParams params{};
    build_prefix_masks(prefix_hex, params);
    build_sha1_state(tpl, params);

    const Backend* be = select_backend(backend_override);
    if (!be) return std::nullopt;

    const unsigned nthreads = pick_thread_count();
    fprintf(stderr, "Device         CPU (%u threads, %s)\n\n", nthreads, be->name);

    std::atomic<bool> global_found{false};
    std::vector<WorkerResult> results(nthreads);
    std::vector<std::thread>  threads(nthreads);

    const auto t0 = std::chrono::steady_clock::now();
    const uint64_t per_thread = kMaxSalt / nthreads;
    for (unsigned t = 0; t < nthreads; ++t) {
        const auto align = [](uint64_t v) { return (v + kSimdAlign - 1) & ~(kSimdAlign - 1); };
        const uint64_t start = align(uint64_t(t)     * per_thread);
        const uint64_t end   = (t == nthreads - 1) ? kMaxSalt : align(uint64_t(t+1) * per_thread);
        threads[t] = std::thread(be->fn, std::cref(params), start, end,
                                 std::ref(global_found), std::ref(results[t]));
    }

    print_progress(results, global_found, t0);
    for (auto& th : threads) th.join();

    // Earliest hit wins; if none found, bail.
    const WorkerResult* winner = nullptr;
    uint64_t total_hashes = 0;
    for (auto& r : results) {
        total_hashes += r.hashes;
        if (r.found && (!winner || r.salt < winner->salt)) winner = &r;
    }
    if (!winner) return std::nullopt;

    SolveResult sr;
    auto out = tpl;
    out.set_salt(winner->salt);
    sr.payload = out.payload();
    static constexpr char kHex[] = "0123456789abcdef";
    for (int i = 0; i < 40; ++i)
        sr.hash[i] = kHex[(winner->hash[i / 8] >> ((7 - i % 8) * 4)) & 0xF];

    const double secs = std::chrono::duration<double>(
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
