// cpu_solver.cpp — multi-threaded SHA1 brute-force using x86 SHA-NI
//
// Each worker thread owns a contiguous salt range and runs autonomously.
// No barriers, no batching, no shared counters. The only cross-thread
// synchronization is a single atomic<bool> for early termination.
// Scales from 4 cores to 192+.
//
// Per-hash optimizations:
//   - Salt encoded via nibble LUT directly to big-endian SHA1 words
//   - Incremental: MSG0/MSG1 recomputed every 65536 salts, MSG2 every iter
//   - 4-way interleaved SHA1 to fully saturate SHA-NI pipeline
//     (sha1rnds4 has 5c latency / 1c throughput — need 4+ streams)
//   - Pre-shuffled mask/target to avoid per-hash ABCD shuffle
//   - Deferred E computation: only finalize E on prefix match (~0.01% of hashes)
//   - Per-thread hash counters (no atomic contention for progress)

#include "solver.h"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <openssl/sha.h>
#include <thread>
#include <vector>
#ifdef __linux__
#include <sched.h>
#endif

static constexpr int kSaltBytes = 48;

// ---------------------------------------------------------------------------
// Salt encoding LUT — each nibble (4 bits) → one big-endian 32-bit word
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// Precompute
// ---------------------------------------------------------------------------

struct CPUParams {
    uint32_t prefix_len;
    uint32_t pre_state[5];
    uint32_t prefix_words[5];
    uint32_t mask0, target0;
    uint32_t mask1, target1;
    // Pre-shuffled mask/target in SHA-NI internal DCBA order (avoids per-hash shuffle)
    uint32_t mask0_shuf, target0_shuf;   // matches element 3 of un-shuffled ABCD
    uint32_t mask1_shuf, target1_shuf;   // matches element 2 of un-shuffled ABCD
    __m128i msg3_const;
    alignas(16) uint8_t block[64];
    int salt_off_in_block;
    bool salt_at_block_start;
};

__attribute__((target("sha,sse4.1,ssse3")))
static CPUParams precompute(const ObjectTemplate& tpl, const std::string& prefix_hex) {
    CPUParams p{};
    p.prefix_len = static_cast<uint32_t>(prefix_hex.size());

    auto prefix = hex_string_to_bytes(prefix_hex);
    for (size_t i = 0; i < prefix.size(); ++i)
        p.prefix_words[i / 4] |= uint32_t(prefix[i]) << ((3 - (i % 4)) * 8);

    uint32_t pn = p.prefix_len;
    p.mask0 = (pn >= 8) ? 0xFFFFFFFFu : (0xFFFFFFFFu << ((8u - pn) * 4u));
    p.target0 = p.prefix_words[0] & p.mask0;
    if (pn > 8) {
        uint32_t rem = pn - 8;
        p.mask1 = (rem >= 8) ? 0xFFFFFFFFu : (0xFFFFFFFFu << ((8u - rem) * 4u));
        p.target1 = p.prefix_words[1] & p.mask1;
    }

    // SHA-NI stores ABCD in DCBA order internally (reversed by 0x1B shuffle).
    // Pre-shuffle mask/target so we can check directly without shuffling ABCD.
    // In the un-shuffled register: element 3 = A (h0), element 2 = B (h1).
    p.mask0_shuf = p.mask0;
    p.target0_shuf = p.target0;
    p.mask1_shuf = p.mask1;
    p.target1_shuf = p.target1;

    uint32_t total = static_cast<uint32_t>(tpl.bytes.size());
    int bb = (tpl.salt_offset / 64) * 64;

    SHA_CTX ctx;
    SHA1_Init(&ctx);
    if (bb > 0) SHA1_Update(&ctx, tpl.bytes.data(), bb);
    p.pre_state[0] = ctx.h0; p.pre_state[1] = ctx.h1;
    p.pre_state[2] = ctx.h2; p.pre_state[3] = ctx.h3; p.pre_state[4] = ctx.h4;

    memset(p.block, 0, 64);
    uint32_t avail = total - bb;
    memcpy(p.block, tpl.bytes.data() + bb, avail);
    p.salt_off_in_block = tpl.salt_offset - bb;
    memset(p.block + p.salt_off_in_block, 0, kSaltBytes);
    p.block[avail] = 0x80;
    uint64_t bits = uint64_t(total) * 8;
    for (int i = 0; i < 8; ++i) p.block[63 - i] = uint8_t(bits >> (i * 8));

    p.salt_at_block_start = (p.salt_off_in_block == 0);

    uint32_t tw[4];
    for (int i = 0; i < 4; ++i) {
        int o = (12 + i) * 4;
        tw[i] = (uint32_t(p.block[o]) << 24) | (uint32_t(p.block[o+1]) << 16) |
                 (uint32_t(p.block[o+2]) << 8) | uint32_t(p.block[o+3]);
    }
    p.msg3_const = _mm_set_epi32((int)tw[0], (int)tw[1], (int)tw[2], (int)tw[3]);

    return p;
}

// ---------------------------------------------------------------------------
// SHA1 block from raw bytes (fallback for non-aligned salt offset)
// ---------------------------------------------------------------------------

__attribute__((target("sha,sse4.1,ssse3")))
static void sha1_block_shani(const uint8_t data[64], const uint32_t init[5],
                              uint32_t out[5]) {
    const __m128i MASK = _mm_set_epi64x(0x0001020304050607ULL, 0x08090a0b0c0d0e0fULL);
    __m128i ABCD = _mm_shuffle_epi32(
        _mm_setr_epi32((int)init[0],(int)init[1],(int)init[2],(int)init[3]), 0x1B);
    __m128i E0 = _mm_set_epi32((int)init[4], 0, 0, 0);
    __m128i AS = ABCD, ES = E0, E1;
    __m128i M0 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(data+ 0)), MASK);
    __m128i M1 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(data+16)), MASK);
    __m128i M2 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(data+32)), MASK);
    __m128i M3 = _mm_shuffle_epi8(_mm_loadu_si128((const __m128i*)(data+48)), MASK);

    E0=_mm_add_epi32(E0,M0);E1=ABCD;ABCD=_mm_sha1rnds4_epu32(ABCD,E0,0);
    E1=_mm_sha1nexte_epu32(E1,M1);E0=ABCD;ABCD=_mm_sha1rnds4_epu32(ABCD,E1,0);M0=_mm_sha1msg1_epu32(M0,M1);
    E0=_mm_sha1nexte_epu32(E0,M2);E1=ABCD;ABCD=_mm_sha1rnds4_epu32(ABCD,E0,0);M1=_mm_sha1msg1_epu32(M1,M2);M0=_mm_xor_si128(M0,M2);
    E1=_mm_sha1nexte_epu32(E1,M3);E0=ABCD;M0=_mm_sha1msg2_epu32(M0,M3);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,0);M2=_mm_sha1msg1_epu32(M2,M3);M1=_mm_xor_si128(M1,M3);
    E0=_mm_sha1nexte_epu32(E0,M0);E1=ABCD;M1=_mm_sha1msg2_epu32(M1,M0);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,0);M3=_mm_sha1msg1_epu32(M3,M0);M2=_mm_xor_si128(M2,M0);
    E1=_mm_sha1nexte_epu32(E1,M1);E0=ABCD;M2=_mm_sha1msg2_epu32(M2,M1);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,1);M0=_mm_sha1msg1_epu32(M0,M1);M3=_mm_xor_si128(M3,M1);
    E0=_mm_sha1nexte_epu32(E0,M2);E1=ABCD;M3=_mm_sha1msg2_epu32(M3,M2);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,1);M1=_mm_sha1msg1_epu32(M1,M2);M0=_mm_xor_si128(M0,M2);
    E1=_mm_sha1nexte_epu32(E1,M3);E0=ABCD;M0=_mm_sha1msg2_epu32(M0,M3);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,1);M2=_mm_sha1msg1_epu32(M2,M3);M1=_mm_xor_si128(M1,M3);
    E0=_mm_sha1nexte_epu32(E0,M0);E1=ABCD;M1=_mm_sha1msg2_epu32(M1,M0);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,1);M3=_mm_sha1msg1_epu32(M3,M0);M2=_mm_xor_si128(M2,M0);
    E1=_mm_sha1nexte_epu32(E1,M1);E0=ABCD;M2=_mm_sha1msg2_epu32(M2,M1);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,1);M0=_mm_sha1msg1_epu32(M0,M1);M3=_mm_xor_si128(M3,M1);
    E0=_mm_sha1nexte_epu32(E0,M2);E1=ABCD;M3=_mm_sha1msg2_epu32(M3,M2);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,2);M1=_mm_sha1msg1_epu32(M1,M2);M0=_mm_xor_si128(M0,M2);
    E1=_mm_sha1nexte_epu32(E1,M3);E0=ABCD;M0=_mm_sha1msg2_epu32(M0,M3);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,2);M2=_mm_sha1msg1_epu32(M2,M3);M1=_mm_xor_si128(M1,M3);
    E0=_mm_sha1nexte_epu32(E0,M0);E1=ABCD;M1=_mm_sha1msg2_epu32(M1,M0);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,2);M3=_mm_sha1msg1_epu32(M3,M0);M2=_mm_xor_si128(M2,M0);
    E1=_mm_sha1nexte_epu32(E1,M1);E0=ABCD;M2=_mm_sha1msg2_epu32(M2,M1);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,2);M0=_mm_sha1msg1_epu32(M0,M1);M3=_mm_xor_si128(M3,M1);
    E0=_mm_sha1nexte_epu32(E0,M2);E1=ABCD;M3=_mm_sha1msg2_epu32(M3,M2);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,2);M1=_mm_sha1msg1_epu32(M1,M2);M0=_mm_xor_si128(M0,M2);
    E1=_mm_sha1nexte_epu32(E1,M3);E0=ABCD;M0=_mm_sha1msg2_epu32(M0,M3);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,3);M2=_mm_sha1msg1_epu32(M2,M3);M1=_mm_xor_si128(M1,M3);
    E0=_mm_sha1nexte_epu32(E0,M0);E1=ABCD;M1=_mm_sha1msg2_epu32(M1,M0);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,3);M3=_mm_sha1msg1_epu32(M3,M0);M2=_mm_xor_si128(M2,M0);
    E1=_mm_sha1nexte_epu32(E1,M1);E0=ABCD;M2=_mm_sha1msg2_epu32(M2,M1);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,3);M3=_mm_xor_si128(M3,M1);
    E0=_mm_sha1nexte_epu32(E0,M2);E1=ABCD;M3=_mm_sha1msg2_epu32(M3,M2);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,3);
    E1=_mm_sha1nexte_epu32(E1,M3);E0=ABCD;ABCD=_mm_sha1rnds4_epu32(ABCD,E1,3);
    E0=_mm_sha1nexte_epu32(E0,ES);ABCD=_mm_add_epi32(ABCD,AS);
    ABCD=_mm_shuffle_epi32(ABCD,0x1B);
    _mm_storeu_si128((__m128i*)out,ABCD);
    out[4]=(uint32_t)_mm_extract_epi32(E0,3);
}

// ---------------------------------------------------------------------------
// 4-way interleaved SHA1 round macros.
//
// SHA-NI sha1rnds4 has 5-cycle latency but 1-cycle throughput.
// With 4 independent hash streams (a,b,c,d), we issue a sha1rnds4 every
// cycle across streams, completely hiding the latency pipeline bubble.
// This is the single biggest throughput win over the previous 2-way design.
//
// P is the stream prefix: a, b, c, or d.
// ---------------------------------------------------------------------------

// Rounds 0-3: init E, first rnds4
#define R0_3(P)   P##E=_mm_add_epi32(P##E,P##M0);P##E1=P##A;P##A=_mm_sha1rnds4_epu32(P##A,P##E,0);
// Rounds 4-7
#define R4_7(P)   P##E1=_mm_sha1nexte_epu32(P##E1,P##M1);P##E=P##A;P##A=_mm_sha1rnds4_epu32(P##A,P##E1,0);P##M0=_mm_sha1msg1_epu32(P##M0,P##M1);
// Rounds 8-11
#define R8_11(P)  P##E=_mm_sha1nexte_epu32(P##E,P##M2);P##E1=P##A;P##A=_mm_sha1rnds4_epu32(P##A,P##E,0);P##M1=_mm_sha1msg1_epu32(P##M1,P##M2);P##M0=_mm_xor_si128(P##M0,P##M2);
// Rounds 12-15
#define R12_15(P) P##E1=_mm_sha1nexte_epu32(P##E1,P##M3);P##E=P##A;P##M0=_mm_sha1msg2_epu32(P##M0,P##M3);P##A=_mm_sha1rnds4_epu32(P##A,P##E1,0);P##M2=_mm_sha1msg1_epu32(P##M2,P##M3);P##M1=_mm_xor_si128(P##M1,P##M3);
// Generic even/odd round groups with msg schedule
#define RE(P,fn) P##E=_mm_sha1nexte_epu32(P##E,P##M0);P##E1=P##A;P##M1=_mm_sha1msg2_epu32(P##M1,P##M0);P##A=_mm_sha1rnds4_epu32(P##A,P##E,fn);P##M3=_mm_sha1msg1_epu32(P##M3,P##M0);P##M2=_mm_xor_si128(P##M2,P##M0);
#define RO(P,fn) P##E1=_mm_sha1nexte_epu32(P##E1,P##M1);P##E=P##A;P##M2=_mm_sha1msg2_epu32(P##M2,P##M1);P##A=_mm_sha1rnds4_epu32(P##A,P##E1,fn);P##M0=_mm_sha1msg1_epu32(P##M0,P##M1);P##M3=_mm_xor_si128(P##M3,P##M1);
#define RE2(P,fn) P##E=_mm_sha1nexte_epu32(P##E,P##M2);P##E1=P##A;P##M3=_mm_sha1msg2_epu32(P##M3,P##M2);P##A=_mm_sha1rnds4_epu32(P##A,P##E,fn);P##M1=_mm_sha1msg1_epu32(P##M1,P##M2);P##M0=_mm_xor_si128(P##M0,P##M2);
#define RO2(P,fn) P##E1=_mm_sha1nexte_epu32(P##E1,P##M3);P##E=P##A;P##M0=_mm_sha1msg2_epu32(P##M0,P##M3);P##A=_mm_sha1rnds4_epu32(P##A,P##E1,fn);P##M2=_mm_sha1msg1_epu32(P##M2,P##M3);P##M1=_mm_xor_si128(P##M1,P##M3);
// Final rounds (no more msg schedule needed)
#define R64(P) P##E=_mm_sha1nexte_epu32(P##E,P##M0);P##E1=P##A;P##M1=_mm_sha1msg2_epu32(P##M1,P##M0);P##A=_mm_sha1rnds4_epu32(P##A,P##E,3);P##M3=_mm_sha1msg1_epu32(P##M3,P##M0);P##M2=_mm_xor_si128(P##M2,P##M0);
#define R68(P) P##E1=_mm_sha1nexte_epu32(P##E1,P##M1);P##E=P##A;P##M2=_mm_sha1msg2_epu32(P##M2,P##M1);P##A=_mm_sha1rnds4_epu32(P##A,P##E1,3);P##M3=_mm_xor_si128(P##M3,P##M1);
#define R72(P) P##E=_mm_sha1nexte_epu32(P##E,P##M2);P##E1=P##A;P##M3=_mm_sha1msg2_epu32(P##M3,P##M2);P##A=_mm_sha1rnds4_epu32(P##A,P##E,3);
#define R76(P) P##E1=_mm_sha1nexte_epu32(P##E1,P##M3);P##E=P##A;P##A=_mm_sha1rnds4_epu32(P##A,P##E1,3);
// Finalize ABCD only (deferred E — only compute E on prefix match)
#define RFIN_ABCD(P) P##A=_mm_add_epi32(P##A,ABCD_INIT);

// SHA1 rounds macro for single-hash path (leftover)
#define SHA1_80(M0,M1,M2,M3,AI,EI,AS,ES,ABCD,E0,E1) \
    ABCD=AI;E0=EI;AS=ABCD;ES=E0; \
    E0=_mm_add_epi32(E0,M0);E1=ABCD;ABCD=_mm_sha1rnds4_epu32(ABCD,E0,0); \
    E1=_mm_sha1nexte_epu32(E1,M1);E0=ABCD;ABCD=_mm_sha1rnds4_epu32(ABCD,E1,0);M0=_mm_sha1msg1_epu32(M0,M1); \
    E0=_mm_sha1nexte_epu32(E0,M2);E1=ABCD;ABCD=_mm_sha1rnds4_epu32(ABCD,E0,0);M1=_mm_sha1msg1_epu32(M1,M2);M0=_mm_xor_si128(M0,M2); \
    E1=_mm_sha1nexte_epu32(E1,M3);E0=ABCD;M0=_mm_sha1msg2_epu32(M0,M3);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,0);M2=_mm_sha1msg1_epu32(M2,M3);M1=_mm_xor_si128(M1,M3); \
    E0=_mm_sha1nexte_epu32(E0,M0);E1=ABCD;M1=_mm_sha1msg2_epu32(M1,M0);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,0);M3=_mm_sha1msg1_epu32(M3,M0);M2=_mm_xor_si128(M2,M0); \
    E1=_mm_sha1nexte_epu32(E1,M1);E0=ABCD;M2=_mm_sha1msg2_epu32(M2,M1);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,1);M0=_mm_sha1msg1_epu32(M0,M1);M3=_mm_xor_si128(M3,M1); \
    E0=_mm_sha1nexte_epu32(E0,M2);E1=ABCD;M3=_mm_sha1msg2_epu32(M3,M2);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,1);M1=_mm_sha1msg1_epu32(M1,M2);M0=_mm_xor_si128(M0,M2); \
    E1=_mm_sha1nexte_epu32(E1,M3);E0=ABCD;M0=_mm_sha1msg2_epu32(M0,M3);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,1);M2=_mm_sha1msg1_epu32(M2,M3);M1=_mm_xor_si128(M1,M3); \
    E0=_mm_sha1nexte_epu32(E0,M0);E1=ABCD;M1=_mm_sha1msg2_epu32(M1,M0);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,1);M3=_mm_sha1msg1_epu32(M3,M0);M2=_mm_xor_si128(M2,M0); \
    E1=_mm_sha1nexte_epu32(E1,M1);E0=ABCD;M2=_mm_sha1msg2_epu32(M2,M1);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,1);M0=_mm_sha1msg1_epu32(M0,M1);M3=_mm_xor_si128(M3,M1); \
    E0=_mm_sha1nexte_epu32(E0,M2);E1=ABCD;M3=_mm_sha1msg2_epu32(M3,M2);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,2);M1=_mm_sha1msg1_epu32(M1,M2);M0=_mm_xor_si128(M0,M2); \
    E1=_mm_sha1nexte_epu32(E1,M3);E0=ABCD;M0=_mm_sha1msg2_epu32(M0,M3);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,2);M2=_mm_sha1msg1_epu32(M2,M3);M1=_mm_xor_si128(M1,M3); \
    E0=_mm_sha1nexte_epu32(E0,M0);E1=ABCD;M1=_mm_sha1msg2_epu32(M1,M0);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,2);M3=_mm_sha1msg1_epu32(M3,M0);M2=_mm_xor_si128(M2,M0); \
    E1=_mm_sha1nexte_epu32(E1,M1);E0=ABCD;M2=_mm_sha1msg2_epu32(M2,M1);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,2);M0=_mm_sha1msg1_epu32(M0,M1);M3=_mm_xor_si128(M3,M1); \
    E0=_mm_sha1nexte_epu32(E0,M2);E1=ABCD;M3=_mm_sha1msg2_epu32(M3,M2);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,2);M1=_mm_sha1msg1_epu32(M1,M2);M0=_mm_xor_si128(M0,M2); \
    E1=_mm_sha1nexte_epu32(E1,M3);E0=ABCD;M0=_mm_sha1msg2_epu32(M0,M3);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,3);M2=_mm_sha1msg1_epu32(M2,M3);M1=_mm_xor_si128(M1,M3); \
    E0=_mm_sha1nexte_epu32(E0,M0);E1=ABCD;M1=_mm_sha1msg2_epu32(M1,M0);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,3);M3=_mm_sha1msg1_epu32(M3,M0);M2=_mm_xor_si128(M2,M0); \
    E1=_mm_sha1nexte_epu32(E1,M1);E0=ABCD;M2=_mm_sha1msg2_epu32(M2,M1);ABCD=_mm_sha1rnds4_epu32(ABCD,E1,3);M3=_mm_xor_si128(M3,M1); \
    E0=_mm_sha1nexte_epu32(E0,M2);E1=ABCD;M3=_mm_sha1msg2_epu32(M3,M2);ABCD=_mm_sha1rnds4_epu32(ABCD,E0,3); \
    E1=_mm_sha1nexte_epu32(E1,M3);E0=ABCD;ABCD=_mm_sha1rnds4_epu32(ABCD,E1,3); \
    E0=_mm_sha1nexte_epu32(E0,ES);ABCD=_mm_add_epi32(ABCD,AS);

// ---------------------------------------------------------------------------
// Worker — each thread owns a salt range, runs autonomously until done.
// Progress tracked via per-thread counter in WorkerResult (no atomics).
// ---------------------------------------------------------------------------

struct alignas(64) WorkerResult {
    uint64_t salt;
    uint32_t hash[5];
    bool found;
    volatile uint64_t hashes;  // written by worker, read by main — no atomic needed
    char _pad[64 - sizeof(uint64_t) - sizeof(uint32_t)*5 - sizeof(bool) - sizeof(uint64_t)];
};

// Check prefix match on un-shuffled ABCD (SHA-NI internal DCBA order).
// Element 3 = A (h0), element 2 = B (h1).
// Returns true if prefix matches. If so, fills result and signals global_found.
__attribute__((target("sha,sse4.1,ssse3")))
static inline bool check_and_store(__m128i A, __m128i E_before_fin, __m128i E0_INIT,
                                   __m128i /*ABCD_INIT*/,
                                   uint32_t mask0, uint32_t target0,
                                   uint32_t mask1, uint32_t target1,
                                   uint32_t pn, uint64_t salt,
                                   uint64_t salt_start,
                                   std::atomic<bool>& global_found,
                                   WorkerResult& result) {
    // ABCD is in DCBA order. Shuffle to ABCD for output.
    __m128i ABCD_out = _mm_shuffle_epi32(A, 0x1B);
    uint32_t h0 = (uint32_t)_mm_extract_epi32(ABCD_out, 0);
    if (__builtin_expect((h0 & mask0) == target0, 0)) {
        if (pn <= 8 || ((uint32_t)_mm_extract_epi32(ABCD_out, 1) & mask1) == target1) {
            // Deferred E finalization — only on match
            __m128i E_final = _mm_sha1nexte_epu32(E_before_fin, E0_INIT);
            result.salt = salt;
            _mm_storeu_si128((__m128i*)result.hash, ABCD_out);
            result.hash[4] = (uint32_t)_mm_extract_epi32(E_final, 3);
            result.found = true;
            result.hashes = salt - salt_start + 1;
            global_found.store(true, std::memory_order_relaxed);
            return true;
        }
    }
    return false;
}

__attribute__((target("sha,sse4.1,ssse3")))
static void worker(const CPUParams& p, uint64_t salt_start, uint64_t salt_end,
                   std::atomic<bool>& global_found, WorkerResult& result) {
    result.found = false;
    result.hashes = 0;

    if (p.salt_at_block_start) {
        const __m128i ABCD_INIT = _mm_shuffle_epi32(
            _mm_setr_epi32((int)p.pre_state[0], (int)p.pre_state[1],
                           (int)p.pre_state[2], (int)p.pre_state[3]), 0x1B);
        const __m128i E0_INIT = _mm_set_epi32((int)p.pre_state[4], 0, 0, 0);
        const __m128i MSG3_C = p.msg3_const;
        const uint32_t mask0 = p.mask0, target0 = p.target0;
        const uint32_t mask1 = p.mask1, target1 = p.target1;
        const uint32_t pn = p.prefix_len;

        uint64_t salt = salt_start;
        while (salt < salt_end) {
            __m128i M0_base = _mm_set_epi32(
                (int)salt_lut[(salt >> 44) & 0xF], (int)salt_lut[(salt >> 40) & 0xF],
                (int)salt_lut[(salt >> 36) & 0xF], (int)salt_lut[(salt >> 32) & 0xF]);
            __m128i M1_base = _mm_set_epi32(
                (int)salt_lut[(salt >> 28) & 0xF], (int)salt_lut[(salt >> 24) & 0xF],
                (int)salt_lut[(salt >> 20) & 0xF], (int)salt_lut[(salt >> 16) & 0xF]);

            uint64_t inner_end = (salt | 0xFFFF) + 1;
            if (inner_end > salt_end) inner_end = salt_end;

            if (global_found.load(std::memory_order_relaxed)) return;

            // 4-way interleaved: process 4 salts simultaneously to fully
            // saturate the SHA-NI pipeline (5c latency / 1c throughput).
            uint64_t s = salt;
            uint64_t quad_end = s + ((inner_end - s) & ~3ULL);

            for (; s < quad_end; s += 4) {
                // Set up 4 independent message streams — only M2 differs per salt
                __m128i aM0=M0_base,aM1=M1_base,aM3=MSG3_C;
                __m128i bM0=M0_base,bM1=M1_base,bM3=MSG3_C;
                __m128i cM0=M0_base,cM1=M1_base,cM3=MSG3_C;
                __m128i dM0=M0_base,dM1=M1_base,dM3=MSG3_C;

                __m128i aM2=_mm_set_epi32((int)salt_lut[(s>>12)&0xF],(int)salt_lut[(s>>8)&0xF],(int)salt_lut[(s>>4)&0xF],(int)salt_lut[s&0xF]);
                uint64_t s1=s+1;
                __m128i bM2=_mm_set_epi32((int)salt_lut[(s1>>12)&0xF],(int)salt_lut[(s1>>8)&0xF],(int)salt_lut[(s1>>4)&0xF],(int)salt_lut[s1&0xF]);
                uint64_t s2=s+2;
                __m128i cM2=_mm_set_epi32((int)salt_lut[(s2>>12)&0xF],(int)salt_lut[(s2>>8)&0xF],(int)salt_lut[(s2>>4)&0xF],(int)salt_lut[s2&0xF]);
                uint64_t s3=s+3;
                __m128i dM2=_mm_set_epi32((int)salt_lut[(s3>>12)&0xF],(int)salt_lut[(s3>>8)&0xF],(int)salt_lut[(s3>>4)&0xF],(int)salt_lut[s3&0xF]);

                __m128i aA=ABCD_INIT,aE=E0_INIT,aE1;
                __m128i bA=ABCD_INIT,bE=E0_INIT,bE1;
                __m128i cA=ABCD_INIT,cE=E0_INIT,cE1;
                __m128i dA=ABCD_INIT,dE=E0_INIT,dE1;

                // Rounds 0-15
                R0_3(a) R0_3(b) R0_3(c) R0_3(d)
                R4_7(a) R4_7(b) R4_7(c) R4_7(d)
                R8_11(a) R8_11(b) R8_11(c) R8_11(d)
                R12_15(a) R12_15(b) R12_15(c) R12_15(d)
                // Rounds 16-31 (fn changes 0→1 at round 20)
                RE(a,0) RE(b,0) RE(c,0) RE(d,0)
                RO(a,1) RO(b,1) RO(c,1) RO(d,1)
                RE2(a,1) RE2(b,1) RE2(c,1) RE2(d,1)
                RO2(a,1) RO2(b,1) RO2(c,1) RO2(d,1)
                // Rounds 32-47 (fn changes 1→2 at round 40)
                RE(a,1) RE(b,1) RE(c,1) RE(d,1)
                RO(a,1) RO(b,1) RO(c,1) RO(d,1)
                RE2(a,2) RE2(b,2) RE2(c,2) RE2(d,2)
                RO2(a,2) RO2(b,2) RO2(c,2) RO2(d,2)
                // Rounds 48-63 (fn changes 2→3 at round 60)
                RE(a,2) RE(b,2) RE(c,2) RE(d,2)
                RO(a,2) RO(b,2) RO(c,2) RO(d,2)
                RE2(a,2) RE2(b,2) RE2(c,2) RE2(d,2)
                RO2(a,3) RO2(b,3) RO2(c,3) RO2(d,3)
                // Rounds 64-79 (final, no more msg schedule)
                R64(a) R64(b) R64(c) R64(d)
                R68(a) R68(b) R68(c) R68(d)
                R72(a) R72(b) R72(c) R72(d)
                R76(a) R76(b) R76(c) R76(d)
                // Finalize ABCD only (defer E computation to match check)
                RFIN_ABCD(a) RFIN_ABCD(b) RFIN_ABCD(c) RFIN_ABCD(d)

                // Check all 4 results. ABCD is still in DCBA order.
                // check_and_store handles shuffle + deferred E finalization.
                if (check_and_store(aA, aE, E0_INIT, ABCD_INIT, mask0, target0, mask1, target1, pn, s,   salt_start, global_found, result)) return;
                if (check_and_store(bA, bE, E0_INIT, ABCD_INIT, mask0, target0, mask1, target1, pn, s+1, salt_start, global_found, result)) return;
                if (check_and_store(cA, cE, E0_INIT, ABCD_INIT, mask0, target0, mask1, target1, pn, s+2, salt_start, global_found, result)) return;
                if (check_and_store(dA, dE, E0_INIT, ABCD_INIT, mask0, target0, mask1, target1, pn, s+3, salt_start, global_found, result)) return;
            }

            // Handle 1-3 leftover salts
            for (; s < inner_end; ++s) {
                __m128i M0=M0_base,M1=M1_base,M3=MSG3_C;
                __m128i M2=_mm_set_epi32((int)salt_lut[(s>>12)&0xF],(int)salt_lut[(s>>8)&0xF],(int)salt_lut[(s>>4)&0xF],(int)salt_lut[s&0xF]);
                __m128i ABCD,E0,E1,AS,ES;
                SHA1_80(M0,M1,M2,M3,ABCD_INIT,E0_INIT,AS,ES,ABCD,E0,E1);
                ABCD=_mm_shuffle_epi32(ABCD,0x1B);
                uint32_t h0=(uint32_t)_mm_extract_epi32(ABCD,0);
                if (__builtin_expect((h0&mask0)==target0,0)) {
                    if (pn<=8||((uint32_t)_mm_extract_epi32(ABCD,1)&mask1)==target1) {
                        result.salt=s;_mm_storeu_si128((__m128i*)result.hash,ABCD);
                        result.hash[4]=(uint32_t)_mm_extract_epi32(E0,3);
                        result.found=true;result.hashes=s-salt_start+1;
                        global_found.store(true,std::memory_order_relaxed);return;
                    }
                }
            }
            salt = s;

            // Update counter every 65536 hashes — main thread reads this for progress
            result.hashes = salt - salt_start;
        }
    } else {
        alignas(16) uint8_t block[64];
        memcpy(block, p.block, 64);
        int soff = p.salt_off_in_block;

        for (uint64_t salt = salt_start; salt < salt_end; ++salt) {
            if (__builtin_expect(((salt - salt_start) & 0xFFFF) == 0, 0)) {
                if (global_found.load(std::memory_order_relaxed)) return;
                result.hashes = salt - salt_start;
            }
            uint64_t s = salt;
            uint8_t* dst = block + soff;
            for (int b = 47; b >= 0; --b) { dst[b] = (s & 1) ? '\t' : ' '; s >>= 1; }

            uint32_t h[5];
            sha1_block_shani(block, p.pre_state, h);
            if (__builtin_expect((h[0] & p.mask0) == p.target0, 0)) {
                if (p.prefix_len <= 8 || (h[1] & p.mask1) == p.target1) {
                    result.salt = salt;
                    memcpy(result.hash, h, sizeof(h));
                    result.found = true;
                    result.hashes = salt - salt_start + 1;
                    global_found.store(true, std::memory_order_relaxed);
                    return;
                }
            }
        }
        result.hashes = salt_end - salt_start;
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

std::optional<SolveResult> solve(const ObjectTemplate& tpl,
                                 const std::string& prefix_hex) {
    CPUParams params = precompute(tpl, prefix_hex);

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

    fprintf(stderr, "Device         CPU (%u threads, SHA-NI, 4-way)\n\n", nthreads);

    std::atomic<bool> global_found{false};
    std::vector<WorkerResult> results(nthreads);
    std::vector<std::thread> threads(nthreads);

    auto t0 = std::chrono::steady_clock::now();

    for (unsigned t = 0; t < nthreads; ++t) {
        uint64_t start = uint64_t(t) * per_thread;
        uint64_t end = (t == nthreads - 1) ? kMaxSalt : start + per_thread;
        threads[t] = std::thread(worker, std::cref(params), start, end,
                                 std::ref(global_found), std::ref(results[t]));
    }

    // Progress: sum per-thread counters (no contention — each on its own cache line)
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
                fprintf(stderr, "⏳ %.2fG hashes | %.2f GH/s | %ds elapsed\n",
                        double(total) / 1e9, double(total) / secs / 1e9, secs);
            }
        }
    }

    for (auto& th : threads) th.join();

    // Sum final hash counts
    uint64_t total_hashes = 0;
    for (unsigned t = 0; t < nthreads; ++t)
        total_hashes += results[t].hashes;

    // Find winner (lowest salt)
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
        "✓ Found        %s\n"
        "Time           %.2fs\n"
        "Throughput     %.2f GH/s\n"
        "Hashes Tried   %.2fG\n",
        std::string(sr.hash.data(), 40).c_str(),
        secs, double(total_hashes) / secs / 1e9, double(total_hashes) / 1e9);
    return sr;
}
