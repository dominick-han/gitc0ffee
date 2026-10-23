#pragma once
// Shared types and constants for the CPU solver backends.

#include <atomic>
#include <cstdint>
#if defined(__x86_64__)
#include <immintrin.h>
#endif

static constexpr int kSaltBytes = 48;

// Salt-nibble lookup. Each 4-bit value maps to one big-endian 32-bit word
// of space (0x20) / tab (0x09) bytes: bit k of the nibble picks tab at
// byte-position k (LSB = position 0), space otherwise.
static constexpr uint32_t salt_lut[16] = {
    0x20202020u, 0x20202009u, 0x20200920u, 0x20200909u,
    0x20092020u, 0x20092009u, 0x20090920u, 0x20090909u,
    0x09202020u, 0x09202009u, 0x09200920u, 0x09200909u,
    0x09092020u, 0x09092009u, 0x09090920u, 0x09090909u,
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
#if defined(__x86_64__)
    __m128i  msg3_const;      // SHA-NI: precomputed MSG3 (words 12-15).
#endif
    uint32_t msg_words[16];   // Full final block as 32-bit words (salt words zeroed).
};

struct WorkerResult {
    // Progress counter — read concurrently by the monitoring thread.
    alignas(64) volatile uint64_t hashes;

    // Written only by the owning worker; read after join.
    alignas(64) uint64_t salt;
    uint32_t hash[5];
    bool     found;
};

using WorkerFn = void(*)(const CPUParams&, uint64_t, uint64_t,
                         std::atomic<bool>&, WorkerResult&);

#if defined(__x86_64__)
void avx512_worker(const CPUParams&, uint64_t, uint64_t, std::atomic<bool>&, WorkerResult&);
void sha_ni_worker(const CPUParams&, uint64_t, uint64_t, std::atomic<bool>&, WorkerResult&);
void avx2_worker  (const CPUParams&, uint64_t, uint64_t, std::atomic<bool>&, WorkerResult&);
#elif defined(__aarch64__)
void arm_sha1_worker(const CPUParams&, uint64_t, uint64_t, std::atomic<bool>&, WorkerResult&);
#endif
// Portable fallback, available on every architecture.
void scalar_worker(const CPUParams&, uint64_t, uint64_t, std::atomic<bool>&, WorkerResult&);

// Reference single-block SHA-1 compression of the final block for `salt`,
// starting from p.pre_state. Writes the finalized digest words to out[5].
void sha1_final_block(const CPUParams& p, uint64_t salt, uint32_t out[5]);
