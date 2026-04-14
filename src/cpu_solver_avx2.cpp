// cpu_solver_avx2.cpp - AVX2 backend: 8-way parallel SHA1

#include "cpu_solver_common.h"
#include <immintrin.h>
#include <atomic>

#ifdef __x86_64__

static inline __m256i avx2_rotl(__m256i x, int n) {
    return _mm256_or_si256(_mm256_slli_epi32(x, n), _mm256_srli_epi32(x, 32 - n));
}
static inline __m256i avx2_ch(__m256i b, __m256i c, __m256i d) {
    return _mm256_xor_si256(_mm256_and_si256(b, _mm256_xor_si256(c, d)), d);
}
static inline __m256i avx2_parity(__m256i b, __m256i c, __m256i d) {
    return _mm256_xor_si256(_mm256_xor_si256(b, c), d);
}
static inline __m256i avx2_maj(__m256i b, __m256i c, __m256i d) {
    return _mm256_or_si256(_mm256_and_si256(b, c),
                           _mm256_and_si256(_mm256_or_si256(b, c), d));
}

#define AVX2_ROUND_W(a, b, c, d, e, f, Warr, idx, Kv) do { \
    __m256i _kw = _mm256_add_epi32(Kv, Warr[idx]); \
    __m256i _t = _mm256_add_epi32(_mm256_add_epi32(avx2_rotl(a, 5), f), \
                                   _mm256_add_epi32(e, _kw)); \
    e = d; d = c; c = avx2_rotl(b, 30); b = a; a = _t; \
} while(0)

#define AVX2_EXPAND(W, i) do { \
    __m256i _x = _mm256_xor_si256(_mm256_xor_si256(W[(i-3)&15], W[(i-8)&15]), W[(i-14)&15]); \
    W[(i)&15] = avx2_rotl(_mm256_xor_si256(_x, W[(i-16)&15]), 1); \
} while(0)

#define AVX2_EROUND(W, i, a, b, c, d, e, fn, Kv) do { \
    AVX2_EXPAND(W, i); \
    __m256i _f = fn(b,c,d); \
    AVX2_ROUND_W(a, b, c, d, e, _f, W, (i)&15, Kv); \
} while(0)

__attribute__((target("avx2")))
void avx2_worker(const CPUParams& p, uint64_t salt_start, uint64_t salt_end,
                         std::atomic<bool>& global_found, WorkerResult& result) {
    result.found = false;
    result.hashes = 0;

    const __m256i INIT_A = _mm256_set1_epi32((int)p.pre_state[0]);
    const __m256i INIT_B = _mm256_set1_epi32((int)p.pre_state[1]);
    const __m256i INIT_C = _mm256_set1_epi32((int)p.pre_state[2]);
    const __m256i INIT_D = _mm256_set1_epi32((int)p.pre_state[3]);
    const __m256i INIT_E = _mm256_set1_epi32((int)p.pre_state[4]);

    const __m256i VK0 = _mm256_set1_epi32((int)SHA1_K[0]);
    const __m256i VK1 = _mm256_set1_epi32((int)SHA1_K[1]);
    const __m256i VK2 = _mm256_set1_epi32((int)SHA1_K[2]);
    const __m256i VK3 = _mm256_set1_epi32((int)SHA1_K[3]);

    const uint32_t cw12 = p.msg_words[12], cw13 = p.msg_words[13],
                   cw14 = p.msg_words[14], cw15 = p.msg_words[15];

    const __m256i vmask0 = _mm256_set1_epi32((int)p.mask0);
    const __m256i vtarget0 = _mm256_set1_epi32((int)p.target0);
    const __m256i vmask1 = _mm256_set1_epi32((int)p.mask1);
    const __m256i vtarget1 = _mm256_set1_epi32((int)p.target1);
    const uint32_t pn = p.prefix_len;

    // AVX2 salt LUT: use vgatherdps with index vector
    const __m256i LANE_IDS = _mm256_setr_epi32(0,1,2,3,4,5,6,7);
    const __m256i NIBBLE_MASK = _mm256_set1_epi32(0xF);

    auto scalar_ch  = [](uint32_t b, uint32_t c, uint32_t d) { return (b & c) | ((~b) & d); };
    auto scalar_round = [](uint32_t a, uint32_t e,
                           uint32_t f, uint32_t w, uint32_t k) -> uint32_t {
        return ((a << 5) | (a >> 27)) + f + e + k + w;
    };

    uint64_t salt = salt_start;

    while (salt < salt_end) {
        uint32_t sw[8];
        sw[0] = salt_lut[(salt >> 44) & 0xF];
        sw[1] = salt_lut[(salt >> 40) & 0xF];
        sw[2] = salt_lut[(salt >> 36) & 0xF];
        sw[3] = salt_lut[(salt >> 32) & 0xF];
        sw[4] = salt_lut[(salt >> 28) & 0xF];
        sw[5] = salt_lut[(salt >> 24) & 0xF];
        sw[6] = salt_lut[(salt >> 20) & 0xF];
        sw[7] = salt_lut[(salt >> 16) & 0xF];

        uint32_t sa = p.pre_state[0], sb = p.pre_state[1], sc = p.pre_state[2],
                 sd = p.pre_state[3], se = p.pre_state[4];
        for (int i = 0; i < 8; ++i) {
            uint32_t f = scalar_ch(sb, sc, sd);
            uint32_t t = scalar_round(sa, se, f, sw[i], SHA1_K[0]);
            se = sd; sd = sc; sc = (sb << 30) | (sb >> 2); sb = sa; sa = t;
        }

        __m256i W0_base = _mm256_set1_epi32((int)sw[0]);
        __m256i W1_base = _mm256_set1_epi32((int)sw[1]);
        __m256i W2_base = _mm256_set1_epi32((int)sw[2]);
        __m256i W3_base = _mm256_set1_epi32((int)sw[3]);
        __m256i W4_base = _mm256_set1_epi32((int)sw[4]);
        __m256i W5_base = _mm256_set1_epi32((int)sw[5]);
        __m256i W6_base = _mm256_set1_epi32((int)sw[6]);
        __m256i W7_base = _mm256_set1_epi32((int)sw[7]);

        uint64_t inner_end = (salt | 0xFFFF) + 1;
        if (inner_end > salt_end) inner_end = salt_end;

        if (global_found.load(std::memory_order_relaxed)) return;

        for (uint64_t s = salt; s + 8 <= inner_end; s += 8) {
            uint32_t sw8  = salt_lut[(s >> 12) & 0xF];
            uint32_t sw9  = salt_lut[(s >> 8) & 0xF];
            uint32_t sw10 = salt_lut[(s >> 4) & 0xF];

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

            uint32_t sw16 = rotl32(cw13 ^ sw8  ^ sw[2] ^ sw[0], 1);
            uint32_t sw17 = rotl32(cw14 ^ sw9  ^ sw[3] ^ sw[1], 1);
            uint32_t sw18 = rotl32(cw15 ^ sw10 ^ sw[4] ^ sw[2], 1);
            uint32_t sw20 = rotl32(sw17 ^ cw12 ^ sw[6] ^ sw[4], 1);
            uint32_t sw21 = rotl32(sw18 ^ cw13 ^ sw[7] ^ sw[5], 1);
            uint32_t sw23 = rotl32(sw20 ^ cw15 ^ sw9   ^ sw[7], 1);
            uint32_t sw24 = rotl32(sw21 ^ sw16 ^ sw10  ^ sw8,   1);

            // AVX2 gather for W[11]: each lane has a different salt nibble
            __m256i sv = _mm256_add_epi32(_mm256_set1_epi32((int)s), LANE_IDS);
            __m256i idx = _mm256_and_si256(sv, NIBBLE_MASK);
            __m256i W11_v = _mm256_i32gather_epi32((const int*)salt_lut, idx, 4);

            alignas(32) __m256i W[16];
            W[0]=W0_base; W[1]=W1_base; W[2]=W2_base; W[3]=W3_base;
            W[4]=W4_base; W[5]=W5_base; W[6]=W6_base; W[7]=W7_base;
            W[8]=_mm256_set1_epi32((int)sw8);
            W[9]=_mm256_set1_epi32((int)sw9);
            W[10]=_mm256_set1_epi32((int)sw10);
            W[11]=W11_v;
            W[12]=_mm256_set1_epi32((int)cw12);
            W[13]=_mm256_set1_epi32((int)cw13);
            W[14]=_mm256_set1_epi32((int)cw14);
            W[15]=_mm256_set1_epi32((int)cw15);

            __m256i a = _mm256_set1_epi32((int)ma);
            __m256i b = _mm256_set1_epi32((int)mb);
            __m256i c = _mm256_set1_epi32((int)mc);
            __m256i d = _mm256_set1_epi32((int)md);
            __m256i e = _mm256_set1_epi32((int)me);
            __m256i f;

            f=avx2_ch(b,c,d); AVX2_ROUND_W(a,b,c,d,e,f,W,11,VK0);
            f=avx2_ch(b,c,d); AVX2_ROUND_W(a,b,c,d,e,f,W,12,VK0);
            f=avx2_ch(b,c,d); AVX2_ROUND_W(a,b,c,d,e,f,W,13,VK0);
            f=avx2_ch(b,c,d); AVX2_ROUND_W(a,b,c,d,e,f,W,14,VK0);
            f=avx2_ch(b,c,d); AVX2_ROUND_W(a,b,c,d,e,f,W,15,VK0);

            W[0]=_mm256_set1_epi32((int)sw16);
            f=avx2_ch(b,c,d); AVX2_ROUND_W(a,b,c,d,e,f,W,0,VK0);
            W[1]=_mm256_set1_epi32((int)sw17);
            f=avx2_ch(b,c,d); AVX2_ROUND_W(a,b,c,d,e,f,W,1,VK0);
            W[2]=_mm256_set1_epi32((int)sw18);
            f=avx2_ch(b,c,d); AVX2_ROUND_W(a,b,c,d,e,f,W,2,VK0);

            AVX2_EROUND(W,19,a,b,c,d,e,avx2_ch,VK0);

            W[4]=_mm256_set1_epi32((int)sw20);
            f=avx2_parity(b,c,d); AVX2_ROUND_W(a,b,c,d,e,f,W,4,VK1);
            W[5]=_mm256_set1_epi32((int)sw21);
            f=avx2_parity(b,c,d); AVX2_ROUND_W(a,b,c,d,e,f,W,5,VK1);

            AVX2_EROUND(W,22,a,b,c,d,e,avx2_parity,VK1);

            W[7]=_mm256_set1_epi32((int)sw23);
            f=avx2_parity(b,c,d); AVX2_ROUND_W(a,b,c,d,e,f,W,7,VK1);
            W[8]=_mm256_set1_epi32((int)sw24);
            f=avx2_parity(b,c,d); AVX2_ROUND_W(a,b,c,d,e,f,W,8,VK1);

            AVX2_EROUND(W,25,a,b,c,d,e,avx2_parity,VK1);
            AVX2_EROUND(W,26,a,b,c,d,e,avx2_parity,VK1);
            AVX2_EROUND(W,27,a,b,c,d,e,avx2_parity,VK1);
            AVX2_EROUND(W,28,a,b,c,d,e,avx2_parity,VK1);
            AVX2_EROUND(W,29,a,b,c,d,e,avx2_parity,VK1);
            AVX2_EROUND(W,30,a,b,c,d,e,avx2_parity,VK1);
            AVX2_EROUND(W,31,a,b,c,d,e,avx2_parity,VK1);
            AVX2_EROUND(W,32,a,b,c,d,e,avx2_parity,VK1);
            AVX2_EROUND(W,33,a,b,c,d,e,avx2_parity,VK1);
            AVX2_EROUND(W,34,a,b,c,d,e,avx2_parity,VK1);
            AVX2_EROUND(W,35,a,b,c,d,e,avx2_parity,VK1);
            AVX2_EROUND(W,36,a,b,c,d,e,avx2_parity,VK1);
            AVX2_EROUND(W,37,a,b,c,d,e,avx2_parity,VK1);
            AVX2_EROUND(W,38,a,b,c,d,e,avx2_parity,VK1);
            AVX2_EROUND(W,39,a,b,c,d,e,avx2_parity,VK1);

            AVX2_EROUND(W,40,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,41,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,42,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,43,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,44,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,45,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,46,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,47,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,48,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,49,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,50,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,51,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,52,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,53,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,54,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,55,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,56,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,57,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,58,a,b,c,d,e,avx2_maj,VK2);
            AVX2_EROUND(W,59,a,b,c,d,e,avx2_maj,VK2);

            AVX2_EROUND(W,60,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,61,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,62,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,63,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,64,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,65,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,66,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,67,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,68,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,69,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,70,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,71,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,72,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,73,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,74,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,75,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,76,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,77,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,78,a,b,c,d,e,avx2_parity,VK3);
            AVX2_EROUND(W,79,a,b,c,d,e,avx2_parity,VK3);

            // Deferred finalization
            a = _mm256_add_epi32(a, INIT_A);
            __m256i cmp0 = _mm256_cmpeq_epi32(_mm256_and_si256(a, vmask0), vtarget0);
            int match = _mm256_movemask_epi8(cmp0);

            if (__builtin_expect(match != 0, 0)) {
                b = _mm256_add_epi32(b, INIT_B);
                if (pn > 8) {
                    __m256i cmp1 = _mm256_cmpeq_epi32(_mm256_and_si256(b, vmask1), vtarget1);
                    match &= _mm256_movemask_epi8(cmp1);
                }
                if (match != 0) {
                    c = _mm256_add_epi32(c, INIT_C);
                    d = _mm256_add_epi32(d, INIT_D);
                    e = _mm256_add_epi32(e, INIT_E);

                    alignas(32) uint32_t ha[8], hb[8], hc[8], hd[8], he[8];
                    _mm256_store_si256((__m256i*)ha, a);
                    _mm256_store_si256((__m256i*)hb, b);
                    _mm256_store_si256((__m256i*)hc, c);
                    _mm256_store_si256((__m256i*)hd, d);
                    _mm256_store_si256((__m256i*)he, e);

                    // Each lane is 4 bytes in movemask, find first set lane
                    int lane = __builtin_ctz(match) / 4;
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

    result.hashes = salt_end - salt_start;
}

#endif // __x86_64__
