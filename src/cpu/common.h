#pragma once
// cpu_solver_common.h - shared types and constants for CPU solver backends

#include <atomic>
#include <cstdint>
#include <immintrin.h>

static constexpr int kSaltBytes = 48;

// Salt encoding LUT - each nibble (4 bits) -> one big-endian 32-bit word
static constexpr uint32_t salt_lut[16] = {
    0x20202020u ^ 0x00000000u, 0x20202020u ^ 0x00000029u,
    0x20202020u ^ 0x00002900u, 0x20202020u ^ 0x00002929u,
    0x20202020u ^ 0x00290000u, 0x20202020u ^ 0x00290029u,
    0x20202020u ^ 0x00292900u, 0x20202020u ^ 0x00292929u,
    0x20202020u ^ 0x29000000u, 0x20202020u ^ 0x29000029u,
    0x20202020u ^ 0x29002900u, 0x20202020u ^ 0x29002929u,
    0x20202020u ^ 0x29290000u, 0x20202020u ^ 0x29290029u,
    0x20202020u ^ 0x29292900u, 0x20202020u ^ 0x29292929u,
};

static constexpr uint32_t SHA1_K[4] = {0x5A827999u, 0x6ED9EBA1u, 0x8F1BBCDCu, 0xCA62C1D6u};

static inline uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

struct CPUParams {
    uint32_t prefix_len;
    uint32_t pre_state[5];
    uint32_t mask0, target0;
    uint32_t mask1, target1;
    __m128i msg3_const;       // SHA-NI: precomputed MSG3 (words 12-15)
    uint32_t msg_words[16];   // AVX2/AVX-512: full block as 32-bit words
};

struct WorkerResult {
    // First cache line: progress counter (read by monitoring thread while worker runs)
    alignas(64) volatile uint64_t hashes;

    // Second cache line: result fields (written only by owning worker, read after join)
    alignas(64) uint64_t salt;
    uint32_t hash[5];
    bool found;
};

using WorkerFn = void(*)(const CPUParams&, uint64_t, uint64_t,
                          std::atomic<bool>&, WorkerResult&);

// Backend worker declarations
void avx512_worker(const CPUParams& p, uint64_t salt_start, uint64_t salt_end,
                   std::atomic<bool>& global_found, WorkerResult& result);
void sha_ni_worker(const CPUParams& p, uint64_t salt_start, uint64_t salt_end,
                  std::atomic<bool>& global_found, WorkerResult& result);
void avx2_worker(const CPUParams& p, uint64_t salt_start, uint64_t salt_end,
                 std::atomic<bool>& global_found, WorkerResult& result);
