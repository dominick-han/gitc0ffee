// cpu_solver.cpp - multi-threaded SHA1 brute-force with runtime dispatch
//
// Two backends:
//   1. AVX-512:  16-way SIMD SHA1 using 512-bit registers (preferred)
//   2. SHA-NI:   4-way interleaved using x86 SHA extensions (fallback)
//
// Runtime CPUID check selects the fastest available path.
//
// Each worker thread owns a contiguous salt range and runs autonomously.
// No barriers, no batching, no shared counters. The only cross-thread
// synchronization is a single atomic<bool> for early termination.
//
// AVX-512 optimizations:
//   - _mm512_rol_epi32 for native rotate (1 uop vs 2 for shift+or)
//   - _mm512_ternarylogic_epi32 for Ch/Parity/Maj (1 uop vs 3-4)
//   - Two-level salt loop: words 0-7 recomputed every 65536 salts,
//     words 8-11 every 16 salts (matches SHA-NI incremental strategy)
//   - Rolling 16-element W window instead of W[80] (no stack spill)
//   - Deferred b/c/d/e finalization (only compute on h0 prefix match)
//   - Fully unrolled 80 rounds with interleaved message schedule

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
#include <cpuid.h>
#endif

static constexpr int kSaltBytes = 48;

// ---------------------------------------------------------------------------
// Salt encoding LUT - each nibble (4 bits) -> one big-endian 32-bit word
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
// CPU feature detection
// ---------------------------------------------------------------------------

enum class CpuBackend { SHA_NI, AVX512 };

static CpuBackend detect_backend() {
#if defined(__x86_64__) && defined(__linux__)
    unsigned eax, ebx, ecx, edx;
    if (__get_cpuid_count(7, 0, &eax, &ebx, &ecx, &edx)) {
        bool avx512f  = (ebx >> 16) & 1;
        bool avx512bw = (ebx >> 30) & 1;
        if (avx512f && avx512bw) return CpuBackend::AVX512;
    }
#endif
    return CpuBackend::SHA_NI;
}

// ---------------------------------------------------------------------------
// Shared precompute
// ---------------------------------------------------------------------------

struct CPUParams {
    uint32_t prefix_len;
    uint32_t pre_state[5];
    uint32_t prefix_words[5];
    uint32_t mask0, target0;
    uint32_t mask1, target1;
    uint32_t mask0_shuf, target0_shuf;
    uint32_t mask1_shuf, target1_shuf;
    __m128i msg3_const;
    alignas(16) uint8_t block[64];
    int salt_off_in_block;
    bool salt_at_block_start;
    uint32_t msg_words[16];
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

    p.mask0_shuf = p.mask0;   p.target0_shuf = p.target0;
    p.mask1_shuf = p.mask1;   p.target1_shuf = p.target1;

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

    for (int i = 0; i < 16; ++i) {
        int o = i * 4;
        p.msg_words[i] = (uint32_t(p.block[o]) << 24) | (uint32_t(p.block[o+1]) << 16) |
                          (uint32_t(p.block[o+2]) << 8) | uint32_t(p.block[o+3]);
    }

    uint32_t tw[4];
    for (int i = 0; i < 4; ++i) {
        int o = (12 + i) * 4;
        tw[i] = (uint32_t(p.block[o]) << 24) | (uint32_t(p.block[o+1]) << 16) |
                 (uint32_t(p.block[o+2]) << 8) | uint32_t(p.block[o+3]);
    }
    p.msg3_const = _mm_set_epi32((int)tw[0], (int)tw[1], (int)tw[2], (int)tw[3]);

    return p;
}

static inline uint32_t rotl32(uint32_t x, int n) {
    return (x << n) | (x >> (32 - n));
}

struct alignas(64) WorkerResult {
    uint64_t salt;
    uint32_t hash[5];
    bool found;
    volatile uint64_t hashes;
    char _pad[64 - sizeof(uint64_t) - sizeof(uint32_t)*5 - sizeof(bool) - sizeof(uint64_t)];
};

// ===========================================================================
// AVX-512 BACKEND: 16-way parallel SHA1 using 512-bit SIMD
//
// Register pressure strategy: store W[16] in an aligned stack array.
// Only 5 state regs (a-e) + 1 temp (f) + current W + K live in zmm.
// The W array lives in L1 cache and reloads are ~4 cycles (pipelined).
// This eliminates the 54+ zmm spills the compiler was generating when
// trying to keep everything in registers.
// ===========================================================================

#ifdef __x86_64__

static constexpr uint32_t SHA1_K[4] = {0x5A827999u, 0x6ED9EBA1u, 0x8F1BBCDCu, 0xCA62C1D6u};

// SHA1 round: reads W[idx] from stack array, adds K, does the round.
#define AVX512_ROUND_W(a, b, c, d, e, f, Warr, idx, Kv) do { \
    __m512i _kw = _mm512_add_epi32(Kv, Warr[idx]); \
    __m512i _t = _mm512_add_epi32(_mm512_add_epi32(_mm512_rol_epi32(a, 5), f), \
                                   _mm512_add_epi32(e, _kw)); \
    e = d; d = c; c = _mm512_rol_epi32(b, 30); b = a; a = _t; \
} while(0)

// Message schedule expansion into W array
#define AVX512_EXPAND(W, i) do { \
    __m512i _x = _mm512_ternarylogic_epi32(W[(i-3)&15], W[(i-8)&15], W[(i-14)&15], 0x96); \
    W[(i)&15] = _mm512_rol_epi32(_mm512_xor_si512(_x, W[(i-16)&15]), 1); \
} while(0)

// Expand + round combined: expand W[i], then do round reading W[(i)&15]
#define AVX512_EROUND(W, i, a, b, c, d, e, fn, Kv) do { \
    AVX512_EXPAND(W, i); \
    __m512i _f = fn(b,c,d); \
    AVX512_ROUND_W(a, b, c, d, e, _f, W, (i)&15, Kv); \
} while(0)

__attribute__((target("avx512f,avx512bw")))
static void avx512_worker(const CPUParams& p, uint64_t salt_start, uint64_t salt_end,
                           std::atomic<bool>& global_found, WorkerResult& result) {
    result.found = false;
    result.hashes = 0;

    if (!p.salt_at_block_start) return;

    const __m512i INIT_A = _mm512_set1_epi32((int)p.pre_state[0]);
    const __m512i INIT_B = _mm512_set1_epi32((int)p.pre_state[1]);
    const __m512i INIT_C = _mm512_set1_epi32((int)p.pre_state[2]);
    const __m512i INIT_D = _mm512_set1_epi32((int)p.pre_state[3]);
    const __m512i INIT_E = _mm512_set1_epi32((int)p.pre_state[4]);

    const __m512i VK0 = _mm512_set1_epi32((int)SHA1_K[0]);
    const __m512i VK1 = _mm512_set1_epi32((int)SHA1_K[1]);
    const __m512i VK2 = _mm512_set1_epi32((int)SHA1_K[2]);
    const __m512i VK3 = _mm512_set1_epi32((int)SHA1_K[3]);

    // Constant tail words
    const uint32_t cw12 = p.msg_words[12], cw13 = p.msg_words[13],
                   cw14 = p.msg_words[14], cw15 = p.msg_words[15];

    const __m512i vmask0 = _mm512_set1_epi32((int)p.mask0);
    const __m512i vtarget0 = _mm512_set1_epi32((int)p.target0);
    const __m512i vmask1 = _mm512_set1_epi32((int)p.mask1);
    const __m512i vtarget1 = _mm512_set1_epi32((int)p.target1);
    const uint32_t pn = p.prefix_len;

    const __m512i SALT_LUT = _mm512_loadu_si512((const __m512i*)salt_lut);
    const __m512i NIBBLE_MASK = _mm512_set1_epi32(0xF);
    const __m512i LANE_IDS = _mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);

    #define CH(b,c,d) _mm512_ternarylogic_epi32(b,c,d,0xCA)
    #define PA(b,c,d) _mm512_ternarylogic_epi32(b,c,d,0x96)
    #define MA(b,c,d) _mm512_ternarylogic_epi32(b,c,d,0xE8)

    uint64_t salt = salt_start;

    // Scalar SHA1 round helper for precomputing uniform rounds
    auto scalar_round = [](uint32_t a, uint32_t e,
                           uint32_t f, uint32_t w, uint32_t k) -> uint32_t {
        return ((a << 5) | (a >> 27)) + f + e + k + w;
    };
    auto scalar_ch  = [](uint32_t b, uint32_t c, uint32_t d) { return (b & c) | ((~b) & d); };

    while (salt < salt_end) {
        // Outer loop: words 0-7 uniform (high nibbles identical for 65536 salts)
        // Since W[0]-W[7] are broadcast (identical across all 16 lanes),
        // rounds 0-7 produce identical state in every lane. Precompute
        // as scalar and broadcast into the inner loop — saves 8 SIMD rounds.
        uint32_t sw[8];
        sw[0] = salt_lut[(salt >> 44) & 0xF];
        sw[1] = salt_lut[(salt >> 40) & 0xF];
        sw[2] = salt_lut[(salt >> 36) & 0xF];
        sw[3] = salt_lut[(salt >> 32) & 0xF];
        sw[4] = salt_lut[(salt >> 28) & 0xF];
        sw[5] = salt_lut[(salt >> 24) & 0xF];
        sw[6] = salt_lut[(salt >> 20) & 0xF];
        sw[7] = salt_lut[(salt >> 16) & 0xF];

        // Precompute rounds 0-7 as scalar (Ch function, K0)
        uint32_t sa = p.pre_state[0], sb = p.pre_state[1], sc = p.pre_state[2],
                 sd = p.pre_state[3], se = p.pre_state[4];
        for (int i = 0; i < 8; ++i) {
            uint32_t f = scalar_ch(sb, sc, sd);
            uint32_t t = scalar_round(sa, se, f, sw[i], SHA1_K[0]);
            se = sd; sd = sc; sc = (sb << 30) | (sb >> 2); sb = sa; sa = t;
        }

        // Still need W[0]-W[7] as SIMD for message schedule expansion in rounds 16+
        __m512i W0_base = _mm512_set1_epi32((int)sw[0]);
        __m512i W1_base = _mm512_set1_epi32((int)sw[1]);
        __m512i W2_base = _mm512_set1_epi32((int)sw[2]);
        __m512i W3_base = _mm512_set1_epi32((int)sw[3]);
        __m512i W4_base = _mm512_set1_epi32((int)sw[4]);
        __m512i W5_base = _mm512_set1_epi32((int)sw[5]);
        __m512i W6_base = _mm512_set1_epi32((int)sw[6]);
        __m512i W7_base = _mm512_set1_epi32((int)sw[7]);

        uint64_t inner_end = (salt | 0xFFFF) + 1;
        if (inner_end > salt_end) inner_end = salt_end;

        if (global_found.load(std::memory_order_relaxed)) return;

        for (uint64_t s = salt; s + 16 <= inner_end; s += 16) {
            // W[8]-W[10] are uniform across all 16 lanes (only bits 0-3 differ,
            // which only affects W[11]). Precompute rounds 8-10 as scalar.
            uint32_t sw8  = salt_lut[(s >> 12) & 0xF];
            uint32_t sw9  = salt_lut[(s >> 8) & 0xF];
            uint32_t sw10 = salt_lut[(s >> 4) & 0xF];

            // Precompute rounds 8-10 as scalar
            uint32_t ma = sa, mb = sb, mc = sc, md = sd, me = se;
            {
                uint32_t f8 = scalar_ch(mb, mc, md);
                uint32_t t8 = scalar_round(ma, me, f8, sw8, SHA1_K[0]);
                me = md; md = mc; mc = (mb << 30) | (mb >> 2); mb = ma; ma = t8;

                uint32_t f9 = scalar_ch(mb, mc, md);
                uint32_t t9 = scalar_round(ma, me, f9, sw9, SHA1_K[0]);
                me = md; md = mc; mc = (mb << 30) | (mb >> 2); mb = ma; ma = t9;

                uint32_t f10 = scalar_ch(mb, mc, md);
                uint32_t t10 = scalar_round(ma, me, f10, sw10, SHA1_K[0]);
                me = md; md = mc; mc = (mb << 30) | (mb >> 2); mb = ma; ma = t10;
            }

            // W[16]-W[18] are also uniform: they depend only on W[0]-W[10],W[12]-W[15]
            // (W[11] first appears in W[19]). Precompute as scalar to skip SIMD expansion.
            // W[i] = rotl(W[i-3]^W[i-8]^W[i-14]^W[i-16], 1)
            uint32_t sw16 = rotl32(p.msg_words[13] ^ sw8  ^ sw[2] ^ sw[0], 1);
            uint32_t sw17 = rotl32(p.msg_words[14] ^ sw9  ^ sw[3] ^ sw[1], 1);
            uint32_t sw18 = rotl32(p.msg_words[15] ^ sw10 ^ sw[4] ^ sw[2], 1);
            // W[20],W[21],W[23],W[24] are also uniform (don't depend on W[11])
            uint32_t sw20 = rotl32(sw17 ^ p.msg_words[12] ^ sw[6] ^ sw[4], 1);
            uint32_t sw21 = rotl32(sw18 ^ p.msg_words[13] ^ sw[7] ^ sw[5], 1);
            uint32_t sw23 = rotl32(sw20 ^ p.msg_words[15] ^ sw9  ^ sw[7], 1);
            uint32_t sw24 = rotl32(sw21 ^ sw16             ^ sw10 ^ sw8,  1);

            // SIMD salt LUT lookup — only W[11] varies per lane
            __m512i sv = _mm512_add_epi32(_mm512_set1_epi32((int)s), LANE_IDS);
            __m512i W11_v = _mm512_permutexvar_epi32(_mm512_and_si512(sv, NIBBLE_MASK), SALT_LUT);

            // W array on stack — needed for message schedule expansion in rounds 16+
            alignas(64) __m512i W[16];
            W[0]=W0_base; W[1]=W1_base; W[2]=W2_base; W[3]=W3_base;
            W[4]=W4_base; W[5]=W5_base; W[6]=W6_base; W[7]=W7_base;
            W[8]=_mm512_set1_epi32((int)sw8);
            W[9]=_mm512_set1_epi32((int)sw9);
            W[10]=_mm512_set1_epi32((int)sw10);
            W[11]=W11_v;
            W[12]=_mm512_set1_epi32((int)cw12);
            W[13]=_mm512_set1_epi32((int)cw13);
            W[14]=_mm512_set1_epi32((int)cw14);
            W[15]=_mm512_set1_epi32((int)cw15);

            // Start from precomputed state after round 10
            __m512i a = _mm512_set1_epi32((int)ma);
            __m512i b = _mm512_set1_epi32((int)mb);
            __m512i c = _mm512_set1_epi32((int)mc);
            __m512i d = _mm512_set1_epi32((int)md);
            __m512i e = _mm512_set1_epi32((int)me);
            __m512i f;

            // Round 11: first SIMD round (W[11] varies per lane)
            f=CH(b,c,d); AVX512_ROUND_W(a,b,c,d,e,f,W,11,VK0);
            f=CH(b,c,d); AVX512_ROUND_W(a,b,c,d,e,f,W,12,VK0);
            f=CH(b,c,d); AVX512_ROUND_W(a,b,c,d,e,f,W,13,VK0);
            f=CH(b,c,d); AVX512_ROUND_W(a,b,c,d,e,f,W,14,VK0);
            f=CH(b,c,d); AVX512_ROUND_W(a,b,c,d,e,f,W,15,VK0);

            // Rounds 16-18: use precomputed uniform W values (skip SIMD expansion)
            W[0] = _mm512_set1_epi32((int)sw16);
            f=CH(b,c,d); AVX512_ROUND_W(a,b,c,d,e,f,W,0,VK0);
            W[1] = _mm512_set1_epi32((int)sw17);
            f=CH(b,c,d); AVX512_ROUND_W(a,b,c,d,e,f,W,1,VK0);
            W[2] = _mm512_set1_epi32((int)sw18);
            f=CH(b,c,d); AVX512_ROUND_W(a,b,c,d,e,f,W,2,VK0);

            // Round 19: W[19] varies per lane (depends on W[11]), must use SIMD expansion
            AVX512_EROUND(W,19,a,b,c,d,e,CH,VK0);

            // Rounds 20-21: inject precomputed uniform W (skip expansion)
            W[4] = _mm512_set1_epi32((int)sw20);
            f=PA(b,c,d); AVX512_ROUND_W(a,b,c,d,e,f,W,4,VK1);
            W[5] = _mm512_set1_epi32((int)sw21);
            f=PA(b,c,d); AVX512_ROUND_W(a,b,c,d,e,f,W,5,VK1);

            // Round 22: W[22] varies (depends on W[19])
            AVX512_EROUND(W,22,a,b,c,d,e,PA,VK1);

            // Rounds 23-24: inject precomputed uniform W
            W[7] = _mm512_set1_epi32((int)sw23);
            f=PA(b,c,d); AVX512_ROUND_W(a,b,c,d,e,f,W,7,VK1);
            W[8] = _mm512_set1_epi32((int)sw24);
            f=PA(b,c,d); AVX512_ROUND_W(a,b,c,d,e,f,W,8,VK1);

            // Rounds 25+: all W values have varying dependencies, full expansion
            AVX512_EROUND(W,25,a,b,c,d,e,PA,VK1);
            AVX512_EROUND(W,26,a,b,c,d,e,PA,VK1);
            AVX512_EROUND(W,27,a,b,c,d,e,PA,VK1);
            AVX512_EROUND(W,28,a,b,c,d,e,PA,VK1);
            AVX512_EROUND(W,29,a,b,c,d,e,PA,VK1);
            AVX512_EROUND(W,30,a,b,c,d,e,PA,VK1);
            AVX512_EROUND(W,31,a,b,c,d,e,PA,VK1);
            AVX512_EROUND(W,32,a,b,c,d,e,PA,VK1);
            AVX512_EROUND(W,33,a,b,c,d,e,PA,VK1);
            AVX512_EROUND(W,34,a,b,c,d,e,PA,VK1);
            AVX512_EROUND(W,35,a,b,c,d,e,PA,VK1);
            AVX512_EROUND(W,36,a,b,c,d,e,PA,VK1);
            AVX512_EROUND(W,37,a,b,c,d,e,PA,VK1);
            AVX512_EROUND(W,38,a,b,c,d,e,PA,VK1);
            AVX512_EROUND(W,39,a,b,c,d,e,PA,VK1);

            // Rounds 40-59: Maj
            AVX512_EROUND(W,40,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,41,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,42,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,43,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,44,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,45,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,46,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,47,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,48,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,49,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,50,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,51,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,52,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,53,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,54,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,55,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,56,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,57,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,58,a,b,c,d,e,MA,VK2);
            AVX512_EROUND(W,59,a,b,c,d,e,MA,VK2);

            // Rounds 60-79: Parity
            AVX512_EROUND(W,60,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,61,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,62,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,63,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,64,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,65,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,66,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,67,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,68,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,69,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,70,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,71,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,72,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,73,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,74,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,75,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,76,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,77,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,78,a,b,c,d,e,PA,VK3);
            AVX512_EROUND(W,79,a,b,c,d,e,PA,VK3);

            // Deferred finalization: only add INIT_A, check h0
            a = _mm512_add_epi32(a, INIT_A);
            __mmask16 match = _mm512_cmpeq_epi32_mask(_mm512_and_si512(a, vmask0), vtarget0);

            if (__builtin_expect(match != 0, 0)) {
                b = _mm512_add_epi32(b, INIT_B);
                if (pn > 8)
                    match &= _mm512_cmpeq_epi32_mask(_mm512_and_si512(b, vmask1), vtarget1);
                if (match != 0) {
                    c = _mm512_add_epi32(c, INIT_C);
                    d = _mm512_add_epi32(d, INIT_D);
                    e = _mm512_add_epi32(e, INIT_E);

                    alignas(64) uint32_t ha[16], hb[16], hc[16], hd[16], he[16];
                    _mm512_store_si512((__m512i*)ha, a);
                    _mm512_store_si512((__m512i*)hb, b);
                    _mm512_store_si512((__m512i*)hc, c);
                    _mm512_store_si512((__m512i*)hd, d);
                    _mm512_store_si512((__m512i*)he, e);

                    int lane = __builtin_ctz(match);
                    result.salt = s + lane;
                    result.hash[0] = ha[lane]; result.hash[1] = hb[lane];
                    result.hash[2] = hc[lane]; result.hash[3] = hd[lane];
                    result.hash[4] = he[lane];
                    result.found = true;
                    result.hashes = s + lane - salt_start + 1;
                    global_found.store(true, std::memory_order_relaxed);
                    return;
                }
            }
        }

        result.hashes = inner_end - salt_start;
        salt = inner_end;
    }

    #undef CH
    #undef PA
    #undef MA

    result.hashes = salt_end - salt_start;
}

#endif // __x86_64__

// ===========================================================================
// SHA-NI BACKEND: 4-way interleaved SHA1 using x86 SHA extensions
// ===========================================================================

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

#define R0_3(P)   P##E=_mm_add_epi32(P##E,P##M0);P##E1=P##A;P##A=_mm_sha1rnds4_epu32(P##A,P##E,0);
#define R4_7(P)   P##E1=_mm_sha1nexte_epu32(P##E1,P##M1);P##E=P##A;P##A=_mm_sha1rnds4_epu32(P##A,P##E1,0);P##M0=_mm_sha1msg1_epu32(P##M0,P##M1);
#define R8_11(P)  P##E=_mm_sha1nexte_epu32(P##E,P##M2);P##E1=P##A;P##A=_mm_sha1rnds4_epu32(P##A,P##E,0);P##M1=_mm_sha1msg1_epu32(P##M1,P##M2);P##M0=_mm_xor_si128(P##M0,P##M2);
#define R12_15(P) P##E1=_mm_sha1nexte_epu32(P##E1,P##M3);P##E=P##A;P##M0=_mm_sha1msg2_epu32(P##M0,P##M3);P##A=_mm_sha1rnds4_epu32(P##A,P##E1,0);P##M2=_mm_sha1msg1_epu32(P##M2,P##M3);P##M1=_mm_xor_si128(P##M1,P##M3);
#define RE(P,fn) P##E=_mm_sha1nexte_epu32(P##E,P##M0);P##E1=P##A;P##M1=_mm_sha1msg2_epu32(P##M1,P##M0);P##A=_mm_sha1rnds4_epu32(P##A,P##E,fn);P##M3=_mm_sha1msg1_epu32(P##M3,P##M0);P##M2=_mm_xor_si128(P##M2,P##M0);
#define RO(P,fn) P##E1=_mm_sha1nexte_epu32(P##E1,P##M1);P##E=P##A;P##M2=_mm_sha1msg2_epu32(P##M2,P##M1);P##A=_mm_sha1rnds4_epu32(P##A,P##E1,fn);P##M0=_mm_sha1msg1_epu32(P##M0,P##M1);P##M3=_mm_xor_si128(P##M3,P##M1);
#define RE2(P,fn) P##E=_mm_sha1nexte_epu32(P##E,P##M2);P##E1=P##A;P##M3=_mm_sha1msg2_epu32(P##M3,P##M2);P##A=_mm_sha1rnds4_epu32(P##A,P##E,fn);P##M1=_mm_sha1msg1_epu32(P##M1,P##M2);P##M0=_mm_xor_si128(P##M0,P##M2);
#define RO2(P,fn) P##E1=_mm_sha1nexte_epu32(P##E1,P##M3);P##E=P##A;P##M0=_mm_sha1msg2_epu32(P##M0,P##M3);P##A=_mm_sha1rnds4_epu32(P##A,P##E1,fn);P##M2=_mm_sha1msg1_epu32(P##M2,P##M3);P##M1=_mm_xor_si128(P##M1,P##M3);
#define R64(P) P##E=_mm_sha1nexte_epu32(P##E,P##M0);P##E1=P##A;P##M1=_mm_sha1msg2_epu32(P##M1,P##M0);P##A=_mm_sha1rnds4_epu32(P##A,P##E,3);P##M3=_mm_sha1msg1_epu32(P##M3,P##M0);P##M2=_mm_xor_si128(P##M2,P##M0);
#define R68(P) P##E1=_mm_sha1nexte_epu32(P##E1,P##M1);P##E=P##A;P##M2=_mm_sha1msg2_epu32(P##M2,P##M1);P##A=_mm_sha1rnds4_epu32(P##A,P##E1,3);P##M3=_mm_xor_si128(P##M3,P##M1);
#define R72(P) P##E=_mm_sha1nexte_epu32(P##E,P##M2);P##E1=P##A;P##M3=_mm_sha1msg2_epu32(P##M3,P##M2);P##A=_mm_sha1rnds4_epu32(P##A,P##E,3);
#define R76(P) P##E1=_mm_sha1nexte_epu32(P##E1,P##M3);P##E=P##A;P##A=_mm_sha1rnds4_epu32(P##A,P##E1,3);
#define RFIN_ABCD(P) P##A=_mm_add_epi32(P##A,ABCD_INIT);

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

__attribute__((target("sha,sse4.1,ssse3")))
static inline bool check_and_store(__m128i A, __m128i E_before_fin, __m128i E0_INIT,
                                   __m128i /*ABCD_INIT*/,
                                   uint32_t mask0, uint32_t target0,
                                   uint32_t mask1, uint32_t target1,
                                   uint32_t pn, uint64_t salt,
                                   uint64_t salt_start,
                                   std::atomic<bool>& global_found,
                                   WorkerResult& result) {
    __m128i ABCD_out = _mm_shuffle_epi32(A, 0x1B);
    uint32_t h0 = (uint32_t)_mm_extract_epi32(ABCD_out, 0);
    if (__builtin_expect((h0 & mask0) == target0, 0)) {
        if (pn <= 8 || ((uint32_t)_mm_extract_epi32(ABCD_out, 1) & mask1) == target1) {
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
static void shani_worker(const CPUParams& p, uint64_t salt_start, uint64_t salt_end,
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

            uint64_t s = salt;
            uint64_t quad_end = s + ((inner_end - s) & ~3ULL);

            for (; s < quad_end; s += 4) {
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

                R0_3(a) R0_3(b) R0_3(c) R0_3(d)
                R4_7(a) R4_7(b) R4_7(c) R4_7(d)
                R8_11(a) R8_11(b) R8_11(c) R8_11(d)
                R12_15(a) R12_15(b) R12_15(c) R12_15(d)
                RE(a,0) RE(b,0) RE(c,0) RE(d,0)
                RO(a,1) RO(b,1) RO(c,1) RO(d,1)
                RE2(a,1) RE2(b,1) RE2(c,1) RE2(d,1)
                RO2(a,1) RO2(b,1) RO2(c,1) RO2(d,1)
                RE(a,1) RE(b,1) RE(c,1) RE(d,1)
                RO(a,1) RO(b,1) RO(c,1) RO(d,1)
                RE2(a,2) RE2(b,2) RE2(c,2) RE2(d,2)
                RO2(a,2) RO2(b,2) RO2(c,2) RO2(d,2)
                RE(a,2) RE(b,2) RE(c,2) RE(d,2)
                RO(a,2) RO(b,2) RO(c,2) RO(d,2)
                RE2(a,2) RE2(b,2) RE2(c,2) RE2(d,2)
                RO2(a,3) RO2(b,3) RO2(c,3) RO2(d,3)
                R64(a) R64(b) R64(c) R64(d)
                R68(a) R68(b) R68(c) R68(d)
                R72(a) R72(b) R72(c) R72(d)
                R76(a) R76(b) R76(c) R76(d)
                RFIN_ABCD(a) RFIN_ABCD(b) RFIN_ABCD(c) RFIN_ABCD(d)

                if (check_and_store(aA, aE, E0_INIT, ABCD_INIT, mask0, target0, mask1, target1, pn, s,   salt_start, global_found, result)) return;
                if (check_and_store(bA, bE, E0_INIT, ABCD_INIT, mask0, target0, mask1, target1, pn, s+1, salt_start, global_found, result)) return;
                if (check_and_store(cA, cE, E0_INIT, ABCD_INIT, mask0, target0, mask1, target1, pn, s+2, salt_start, global_found, result)) return;
                if (check_and_store(dA, dE, E0_INIT, ABCD_INIT, mask0, target0, mask1, target1, pn, s+3, salt_start, global_found, result)) return;
            }

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
// Public API - runtime dispatch between AVX-512 and SHA-NI
// ---------------------------------------------------------------------------

using WorkerFn = void(*)(const CPUParams&, uint64_t, uint64_t,
                          std::atomic<bool>&, WorkerResult&);

std::optional<SolveResult> solve(const ObjectTemplate& tpl,
                                 const std::string& prefix_hex) {
    CPUParams params = precompute(tpl, prefix_hex);

    CpuBackend backend = detect_backend();
    const char* backend_name = "SHA-NI, 4-way";
    WorkerFn worker_fn = shani_worker;

#ifdef __x86_64__
    if (backend == CpuBackend::AVX512 && params.salt_at_block_start) {
        backend_name = "AVX-512, 16-way";
        worker_fn = avx512_worker;
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
