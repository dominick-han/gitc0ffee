// avx2.cpp - AVX2 backend: 8-way parallel SHA-1 brute force
//
// Each SIMD batch hashes 8 salts at once (one per ymm lane). The salt's
// 12 nibbles map one-to-one onto W[0..11] of the SHA-1 message schedule.
// When enumerating salts in order, high nibbles stay fixed for long runs,
// so we can precompute the SHA-1 rounds that only depend on those high
// nibbles once and reuse the partial state across many hashes:
//
//   Level 3 (per 65536 salts)  W[0..7]  constant -> precompute rounds 0-7
//   Level 2 (per  4096 salts)  + W[8]   constant -> precompute round  8
//   Level 1 (per   256 salts)  + W[9]   constant -> precompute round  9
//   Level 0 (per     8 salts)  + W[10]  constant -> precompute round 10
//                                W[11]  differs per lane -> round 11 collapses to
//                                                           a single SIMD add, then
//                                                           rounds 12+ run SIMD

#include "cpu/common.h"
#include <algorithm>
#include <atomic>
#include <immintrin.h>

#pragma GCC target("avx2")

// SHA-1 primitives (SIMD).
static inline __m256i rol   (__m256i x, int n)                 { return _mm256_or_si256(_mm256_slli_epi32(x, n), _mm256_srli_epi32(x, 32 - n)); }
static inline __m256i f_ch  (__m256i b, __m256i c, __m256i d)  { return _mm256_xor_si256(_mm256_and_si256(b, _mm256_xor_si256(c, d)), d); }
static inline __m256i f_par (__m256i b, __m256i c, __m256i d)  { return _mm256_xor_si256(_mm256_xor_si256(b, c), d); }
static inline __m256i f_maj (__m256i b, __m256i c, __m256i d)  { return _mm256_or_si256(_mm256_and_si256(b, c), _mm256_and_si256(_mm256_or_si256(b, c), d)); }

// W[i] = rol(W[i-3] ^ W[i-8] ^ W[i-14] ^ W[i-16], 1)
static inline __m256i expand(__m256i w3, __m256i w8, __m256i w14, __m256i w16) {
    return rol(_mm256_xor_si256(_mm256_xor_si256(w3, w8), _mm256_xor_si256(w14, w16)), 1);
}

// One SHA-1 round: t = rol(a,5) + f + e + k + w; (a,b,c,d,e) <- (t, a, rol(b,30), c, d)
static inline void sha1_round(__m256i& a, __m256i& b, __m256i& c, __m256i& d, __m256i& e, __m256i f, __m256i w, __m256i k) {
    __m256i t = _mm256_add_epi32(_mm256_add_epi32(rol(a, 5), f), _mm256_add_epi32(e, _mm256_add_epi32(k, w)));
    e = d; d = c; c = rol(b, 30); b = a; a = t;
}

// Scalar round, used to precompute the SHA-1 prefix where every lane agrees.
static inline void scalar_round(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, uint32_t& e, uint32_t w, uint32_t k) {
    uint32_t f = (b & c) | (~b & d);
    uint32_t t = rotl32(a, 5) + f + e + k + w;
    e = d; d = c; c = rotl32(b, 30); b = a; a = t;
}

void avx2_worker(const CPUParams& p, uint64_t salt_start, uint64_t salt_end, std::atomic<bool>& global_found, WorkerResult& result) {
    result.found  = false;
    result.hashes = 0;

    const __m256i VK[4] = {
        _mm256_set1_epi32((int)SHA1_K[0]), _mm256_set1_epi32((int)SHA1_K[1]),
        _mm256_set1_epi32((int)SHA1_K[2]), _mm256_set1_epi32((int)SHA1_K[3]),
    };

    // W[11] differs per lane only in its low nibble. Inner-loop salts are
    // 8-aligned, so lane i's nibble is exactly i — W[11] is just the first
    // 8 entries of the salt LUT loaded verbatim.
    const __m256i W11 = _mm256_loadu_si256((const __m256i*)salt_lut);

    // Level 3: per 65536 salts. W[0..7] constant, run rounds 0-7 as scalar.
    for (uint64_t salt = salt_start; salt < salt_end; ) {
        uint32_t sw[11];
        uint32_t l3a = p.pre_state[0], l3b = p.pre_state[1], l3c = p.pre_state[2],
                 l3d = p.pre_state[3], l3e = p.pre_state[4];
        for (int i = 0; i < 8; ++i) {
            sw[i] = salt_lut[(salt >> (44 - 4*i)) & 0xF];
            scalar_round(l3a, l3b, l3c, l3d, l3e, sw[i], SHA1_K[0]);
        }

        // Pre-broadcast every W slot that's constant across this whole block.
        __m256i Wv[16];
        for (int i = 0; i < 8; ++i) Wv[i] = _mm256_set1_epi32((int)sw[i]);
        Wv[11] = W11;
        for (int i = 12; i < 16; ++i) Wv[i] = _mm256_set1_epi32((int)p.msg_words[i]);

        const uint64_t l3_end = std::min<uint64_t>(salt_end, (salt | 0xFFFFull) + 1);
        if (global_found.load(std::memory_order_relaxed)) return;

        // Level 2: per 4096 salts. + W[8] constant, run round 8 as scalar.
        for (uint64_t s3 = salt; s3 < l3_end; s3 = (s3 | 0xFFFull) + 1) {
            sw[8] = salt_lut[(s3 >> 12) & 0xF];
            uint32_t l2a = l3a, l2b = l3b, l2c = l3c, l2d = l3d, l2e = l3e;
            scalar_round(l2a, l2b, l2c, l2d, l2e, sw[8], SHA1_K[0]);
            Wv[8] = _mm256_set1_epi32((int)sw[8]);

            const uint64_t l2_end = std::min<uint64_t>(l3_end, (s3 | 0xFFFull) + 1);

            // Level 1: per 256 salts. + W[9] constant, run round 9 as scalar.
            for (uint64_t s2 = s3; s2 < l2_end; s2 = (s2 | 0xFFull) + 1) {
                sw[9] = salt_lut[(s2 >> 8) & 0xF];
                uint32_t l1a = l2a, l1b = l2b, l1c = l2c, l1d = l2d, l1e = l2e;
                scalar_round(l1a, l1b, l1c, l1d, l1e, sw[9], SHA1_K[0]);
                Wv[9] = _mm256_set1_epi32((int)sw[9]);

                const uint64_t l1_end = std::min<uint64_t>(l2_end, (s2 | 0xFFull) + 1);

                // Level 0: one batch of 8 salts. + W[10] constant, run round 10.
                for (uint64_t s = s2; s + 8 <= l1_end; s += 8) {
                    sw[10] = salt_lut[(s >> 4) & 0xF];
                    uint32_t ia = l1a, ib = l1b, ic = l1c, id = l1d, ie = l1e;
                    scalar_round(ia, ib, ic, id, ie, sw[10], SHA1_K[0]);

                    // Seed the SIMD W window. W[11] is the only per-lane word
                    // (one nibble per lane); the rest are lane-uniform broadcasts.
                    __m256i W[16];
                    for (int i = 0; i < 16; ++i) W[i] = Wv[i];
                    W[10] = _mm256_set1_epi32((int)sw[10]);

                    // Round 11 is partially uniform: only W[11] varies across lanes
                    // while (a,b,c,d,e) are still scalar broadcasts. Fold the entire
                    // scalar part of the round into one value and broadcast it, so
                    // SIMD round 11 collapses from a full sha1_round into a single
                    // vpaddd.
                    //
                    //   t_11 = rol(ia,5) + f_ch(ib,ic,id) + ie + K0 + W[11]
                    //        = broadcast(scalar_sum) + W[11]
                    const uint32_t f11        = (ib & ic) | (~ib & id);
                    const uint32_t t11_scalar = rotl32(ia, 5) + f11 + ie + SHA1_K[0];

                    // Post-round-11 state: only `a` varies per lane; b,c,d,e remain
                    // lane-uniform (shift pattern of sha1_round with a_new = t_11).
                    __m256i a = _mm256_add_epi32(W[11], _mm256_set1_epi32((int)t11_scalar));
                    __m256i b = _mm256_set1_epi32((int)ia);
                    __m256i c = _mm256_set1_epi32((int)rotl32(ib, 30));
                    __m256i d = _mm256_set1_epi32((int)ic);
                    __m256i e = _mm256_set1_epi32((int)id);

                    // Rounds 12-79: full SIMD. Split at round-function boundaries
                    // (20/40/60). Rounds 12-15 use pre-loaded W[12..15]; from
                    // round 16 on, W[] is a circular buffer (W[i] overwrites
                    // W[(i-16)&15] via message expansion).
                    #pragma GCC unroll 4
                    for (int i = 12; i < 16; ++i)
                        sha1_round(a, b, c, d, e, f_ch(b,c,d), W[i], VK[0]);
                    #pragma GCC unroll 4
                    for (int i = 16; i < 20; ++i) {
                        W[i&15] = expand(W[(i-3)&15], W[(i-8)&15], W[(i-14)&15], W[(i-16)&15]);
                        sha1_round(a, b, c, d, e, f_ch(b,c,d), W[i&15], VK[0]);
                    }
                    #pragma GCC unroll 20
                    for (int i = 20; i < 40; ++i) {
                        W[i&15] = expand(W[(i-3)&15], W[(i-8)&15], W[(i-14)&15], W[(i-16)&15]);
                        sha1_round(a, b, c, d, e, f_par(b,c,d), W[i&15], VK[1]);
                    }
                    #pragma GCC unroll 20
                    for (int i = 40; i < 60; ++i) {
                        W[i&15] = expand(W[(i-3)&15], W[(i-8)&15], W[(i-14)&15], W[(i-16)&15]);
                        sha1_round(a, b, c, d, e, f_maj(b,c,d), W[i&15], VK[2]);
                    }
                    #pragma GCC unroll 20
                    for (int i = 60; i < 80; ++i) {
                        W[i&15] = expand(W[(i-3)&15], W[(i-8)&15], W[(i-14)&15], W[(i-16)&15]);
                        sha1_round(a, b, c, d, e, f_par(b,c,d), W[i&15], VK[3]);
                    }

                    // Deferred finalization: only add pre_state[0] and compare the top
                    // prefix nibbles. False positives (~1 in 2^32) fall through to the
                    // cold path, which computes the full hash.
                    __m256i h0 = _mm256_add_epi32(a, _mm256_set1_epi32((int)p.pre_state[0]));
                    int match = _mm256_movemask_epi8(_mm256_cmpeq_epi32(
                        _mm256_and_si256(h0, _mm256_set1_epi32((int)p.mask0)),
                        _mm256_set1_epi32((int)p.target0)));

                    if (__builtin_expect(match == 0, 1)) continue;

                    __m256i h1 = _mm256_add_epi32(b, _mm256_set1_epi32((int)p.pre_state[1]));
                    if (p.prefix_len > 8) {
                        match &= _mm256_movemask_epi8(_mm256_cmpeq_epi32(
                            _mm256_and_si256(h1, _mm256_set1_epi32((int)p.mask1)),
                            _mm256_set1_epi32((int)p.target1)));
                        if (match == 0) continue;
                    }

                    // Full hash extraction (runs at most once per solve).
                    // movemask_epi8 produces 4 bits per 32-bit lane -> divide by 4.
                    int lane = __builtin_ctz(match) / 4;
                    alignas(32) uint32_t buf[8];
                    auto extract = [&](__m256i v) { _mm256_store_si256((__m256i*)buf, v); return buf[lane]; };
                    result.hash[0] = extract(h0);
                    result.hash[1] = extract(h1);
                    result.hash[2] = extract(_mm256_add_epi32(c, _mm256_set1_epi32((int)p.pre_state[2])));
                    result.hash[3] = extract(_mm256_add_epi32(d, _mm256_set1_epi32((int)p.pre_state[3])));
                    result.hash[4] = extract(_mm256_add_epi32(e, _mm256_set1_epi32((int)p.pre_state[4])));
                    result.salt    = s + lane;
                    result.found   = true;
                    result.hashes  = s + lane - salt_start + 1;
                    global_found.store(true, std::memory_order_relaxed);
                    return;
                }
            }
        }
        result.hashes = l3_end - salt_start;
        salt = l3_end;
    }
    result.hashes = salt_end - salt_start;
}
