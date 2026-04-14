// cpu_solver.cpp - multi-threaded SHA1 brute-force with runtime dispatch
//
// Three backends (selected by CPUID, priority order):
//   1. AVX-512:  16-way SIMD SHA1 using 512-bit registers
//   2. SHA-NI:   4-way interleaved using x86 SHA extensions
//   3. AVX2:     8-way SIMD SHA1 using 256-bit registers (universal fallback)

#include "solver.h"
#include "cpu_solver_common.h"

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
// CPU feature detection: AVX-512 > SHA-NI > AVX2
// ---------------------------------------------------------------------------

enum class CpuBackend { AVX512, SHA_NI, AVX2 };

static CpuBackend detect_backend() {
#if defined(__x86_64__) && defined(__linux__)
    unsigned eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        if ((ebx >> 16) & 1 & (ebx >> 30))  // AVX512F + AVX512BW
            return CpuBackend::AVX512;
        if ((ebx >> 29) & 1)                 // SHA-NI
            return CpuBackend::SHA_NI;
    }
#endif
    return CpuBackend::AVX2;
}

// ---------------------------------------------------------------------------
// Shared precompute
// ---------------------------------------------------------------------------

static CPUParams precompute(const ObjectTemplate& tpl, const std::string& prefix_hex) {
    CPUParams p{};
    p.prefix_len = static_cast<uint32_t>(prefix_hex.size());

    auto prefix = hex_string_to_bytes(prefix_hex);
    uint32_t prefix_words[5] = {};
    for (size_t i = 0; i < prefix.size(); ++i)
        prefix_words[i / 4] |= uint32_t(prefix[i]) << ((3 - (i % 4)) * 8);

    uint32_t pn = p.prefix_len;
    p.mask0 = (pn >= 8) ? 0xFFFFFFFFu : (0xFFFFFFFFu << ((8u - pn) * 4u));
    p.target0 = prefix_words[0] & p.mask0;
    if (pn > 8) {
        uint32_t rem = pn - 8;
        p.mask1 = (rem >= 8) ? 0xFFFFFFFFu : (0xFFFFFFFFu << ((8u - rem) * 4u));
        p.target1 = prefix_words[1] & p.mask1;
    }

    uint32_t total = static_cast<uint32_t>(tpl.bytes.size());
    int bb = (tpl.salt_offset / 64) * 64;

    SHA_CTX ctx;
    SHA1_Init(&ctx);
    if (bb > 0) SHA1_Update(&ctx, tpl.bytes.data(), bb);
    p.pre_state[0] = ctx.h0; p.pre_state[1] = ctx.h1;
    p.pre_state[2] = ctx.h2; p.pre_state[3] = ctx.h3; p.pre_state[4] = ctx.h4;

    // Build the final SHA1 block: salt is always at block start
    uint8_t block[64] = {};
    uint32_t avail = total - bb;
    memcpy(block, tpl.bytes.data() + bb, avail);
    memset(block, 0, kSaltBytes);  // zero salt region at offset 0
    block[avail] = 0x80;
    uint64_t bits = uint64_t(total) * 8;
    for (int i = 0; i < 8; ++i) block[63 - i] = uint8_t(bits >> (i * 8));

    for (int i = 0; i < 16; ++i) {
        int o = i * 4;
        p.msg_words[i] = (uint32_t(block[o]) << 24) | (uint32_t(block[o+1]) << 16) |
                          (uint32_t(block[o+2]) << 8) | uint32_t(block[o+3]);
    }

    p.msg3_const = _mm_set_epi32((int)p.msg_words[12], (int)p.msg_words[13],
                                (int)p.msg_words[14], (int)p.msg_words[15]);

    return p;
}

// ---------------------------------------------------------------------------
// Public API - runtime dispatch
// ---------------------------------------------------------------------------

std::optional<SolveResult> solve(const ObjectTemplate& tpl,
                                 const std::string& prefix_hex) {
    CPUParams params = precompute(tpl, prefix_hex);

    CpuBackend backend = detect_backend();
    const char* backend_name = "AVX2, 8-way";
    WorkerFn worker_fn = avx2_worker;

#ifdef __x86_64__
    if (backend == CpuBackend::AVX512) {
        backend_name = "AVX-512, 16-way";
        worker_fn = avx512_worker;
    } else if (backend == CpuBackend::SHA_NI) {
        backend_name = "SHA-NI, 4-way";
        worker_fn = sha_ni_worker;
    }
#endif

    unsigned nthreads = std::thread::hardware_concurrency();
    if (nthreads == 0) nthreads = 4;

#ifdef __linux__
    {
        cpu_set_t cpuset;
        if (sched_getaffinity(0, sizeof(cpuset), &cpuset) == 0) {
            unsigned avail = (unsigned)CPU_COUNT(&cpuset);
            if (avail > 0 && avail < nthreads) nthreads = avail;
        }
    }
#endif

    static constexpr uint64_t kMaxSalt = (1ULL << 48) - 1;
    uint64_t per_thread = kMaxSalt / nthreads;

    fprintf(stderr, "Device         CPU (%u threads, %s)\n\n", nthreads, backend_name);

    std::atomic<bool> global_found{false};
    std::vector<WorkerResult> results(nthreads);
    std::vector<std::thread> threads(nthreads);

    auto t0 = std::chrono::steady_clock::now();

    for (unsigned t = 0; t < nthreads; ++t) {
        uint64_t start = uint64_t(t) * per_thread;
        uint64_t end = (t == nthreads - 1) ? kMaxSalt : start + per_thread;
        threads[t] = std::thread(worker_fn, std::cref(params), start, end,
                                 std::ref(global_found), std::ref(results[t]));
    }

    {
        int next_sec = 1;
        while (!global_found.load(std::memory_order_relaxed)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            auto now = std::chrono::steady_clock::now();
            int secs = int(std::chrono::duration<double>(now - t0).count());
            if (secs >= next_sec) {
                next_sec = secs + 1;
                uint64_t total = 0;
                for (unsigned t = 0; t < nthreads; ++t)
                    total += results[t].hashes;
                fprintf(stderr, "%.2fG hashes | %.2f GH/s | %ds elapsed\n",
                        double(total) / 1e9, double(total) / secs / 1e9, secs);
            }
        }
    }

    for (auto& th : threads) th.join();

    uint64_t total_hashes = 0;
    for (unsigned t = 0; t < nthreads; ++t)
        total_hashes += results[t].hashes;

    int winner = -1;
    for (unsigned t = 0; t < nthreads; ++t) {
        if (results[t].found) {
            if (winner < 0 || results[t].salt < results[(unsigned)winner].salt)
                winner = (int)t;
        }
    }

    if (winner < 0) return std::nullopt;

    auto out = tpl;
    out.set_salt(results[winner].salt);

    SolveResult sr;
    sr.payload = out.payload();
    static constexpr char hx[] = "0123456789abcdef";
    for (int i = 0; i < 5; ++i) {
        uint32_t w = results[winner].hash[i];
        sr.hash[i*8]   = hx[(w>>28)&0xF]; sr.hash[i*8+1] = hx[(w>>24)&0xF];
        sr.hash[i*8+2] = hx[(w>>20)&0xF]; sr.hash[i*8+3] = hx[(w>>16)&0xF];
        sr.hash[i*8+4] = hx[(w>>12)&0xF]; sr.hash[i*8+5] = hx[(w>>8)&0xF];
        sr.hash[i*8+6] = hx[(w>>4)&0xF];  sr.hash[i*8+7] = hx[w&0xF];
    }

    double secs = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    fprintf(stderr,
        "Found          %s\n"
        "Time           %.2fs\n"
        "Throughput     %.2f GH/s\n"
        "Hashes Tried   %.2fG\n",
        std::string(sr.hash.data(), 40).c_str(),
        secs, double(total_hashes) / secs / 1e9, double(total_hashes) / 1e9);
    return sr;
}
