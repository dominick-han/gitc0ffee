// cpu_solver_avx512.cpp - AVX-512 backend: 16-way parallel SHA1

#include "cpu_solver_common.h"
#include <immintrin.h>
#include <atomic>

#ifdef __x86_64__

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
void avx512_worker(const CPUParams& p, uint64_t salt_start, uint64_t salt_end,
                           std::atomic<bool>& global_found, WorkerResult& result) {
    result.found = false;
    result.hashes = 0;

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
