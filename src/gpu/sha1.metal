#include <metal_stdlib>
using namespace metal;

static constant uint32_t K0 = 0x5A827999u;
static constant uint32_t K1 = 0x6ED9EBA1u;
static constant uint32_t K2 = 0x8F1BBCDCu;
static constant uint32_t K3 = 0xCA62C1D6u;

inline uint32_t rotl(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32u - n));
}

static constant uint32_t salt_lut[16] = {
    0x00000000, 0x00000029, 0x00002900, 0x00002929,
    0x00290000, 0x00290029, 0x00292900, 0x00292929,
    0x29000000, 0x29000029, 0x29002900, 0x29002929,
    0x29290000, 0x29290029, 0x29292900, 0x29292929
};

inline uint32_t encode_salt_word(uint64_t salt, int word_idx) {
    return 0x20202020u ^ salt_lut[(salt >> ((11 - word_idx) * 4)) & 0xF];
}

struct GPUParams {
    uint32_t mask0;
    uint32_t target0;
    uint32_t mask1;
    uint32_t target1;
    uint32_t prefix_len;
    uint32_t pre_state[5];
    uint32_t tail_w[4];
    uint32_t _pad;
    uint64_t base_salt;
};

inline uint32_t sha1_ch(uint32_t b, uint32_t c, uint32_t d) {
    return (b & c) ^ ((~b) & d);
}
inline uint32_t sha1_round(uint32_t a, uint32_t e, uint32_t f, uint32_t w, uint32_t k) {
    return rotl(a, 5u) + f + e + k + w;
}

// Threadgroup precomputed state after round 9.
// With threadgroup size 256 aligned to 256, bits 0-7 vary:
//   W[9] (bits 11..8) uniform, W[10] (bits 7..4) varies, W[11] (bits 3..0) varies.
// Thread 0 precomputes rounds 0-9 and caches all params.
struct PrecomputedState {
    uint32_t a, b, c, d, e;
    uint32_t h0_init, h1_init, h2_init, h3_init, h4_init;
    uint32_t sw[10];
    uint32_t cw12, cw13, cw14, cw15;
    uint32_t sw16, sw17, sw20;
    uint32_t mask0, target0, mask1, target1, prefix_len;
};

// Round macros with K fused into W — saves one add per round on critical path
#define RK0(wk) { uint32_t t = rotl(a,5u) + ((b&c)^((~b)&d))     + e + wk; e=d; d=c; c=rotl(b,30u); b=a; a=t; }
#define RK1(wk) { uint32_t t = rotl(a,5u) + (b^c^d)               + e + wk; e=d; d=c; c=rotl(b,30u); b=a; a=t; }
#define RK2(wk) { uint32_t t = rotl(a,5u) + ((b&c)^(b&d)^(c&d))   + e + wk; e=d; d=c; c=rotl(b,30u); b=a; a=t; }
#define RK3(wk) { uint32_t t = rotl(a,5u) + (b^c^d)               + e + wk; e=d; d=c; c=rotl(b,30u); b=a; a=t; }

kernel void bruteforce_sha1(
    constant GPUParams& params [[buffer(0)]],
    device uint8_t*     result [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    device auto* found = reinterpret_cast<device atomic_uint*>(result);
    if (atomic_load_explicit(found, memory_order_relaxed) != 0) return;

    uint64_t salt = params.base_salt + uint64_t(tid);

    threadgroup PrecomputedState sp;

    if (lid == 0) {
        uint64_t gb = salt;
        sp.h0_init = params.pre_state[0]; sp.h1_init = params.pre_state[1];
        sp.h2_init = params.pre_state[2]; sp.h3_init = params.pre_state[3];
        sp.h4_init = params.pre_state[4];
        sp.cw12 = params.tail_w[0]; sp.cw13 = params.tail_w[1];
        sp.cw14 = params.tail_w[2]; sp.cw15 = params.tail_w[3];
        sp.mask0 = params.mask0; sp.target0 = params.target0;
        sp.mask1 = params.mask1; sp.target1 = params.target1;
        sp.prefix_len = params.prefix_len;

        for (int i = 0; i < 10; ++i)
            sp.sw[i] = encode_salt_word(gb, i);

        uint32_t a = sp.h0_init, b = sp.h1_init, c = sp.h2_init,
                 d = sp.h3_init, e = sp.h4_init;
        for (int i = 0; i < 10; ++i) {
            uint32_t f = sha1_ch(b, c, d);
            uint32_t t = sha1_round(a, e, f, sp.sw[i], K0);
            e = d; d = c; c = rotl(b, 30u); b = a; a = t;
        }
        sp.a = a; sp.b = b; sp.c = c; sp.d = d; sp.e = e;

        sp.sw16 = rotl(sp.cw13 ^ sp.sw[8] ^ sp.sw[2] ^ sp.sw[0], 1u);
        sp.sw17 = rotl(sp.cw14 ^ sp.sw[9] ^ sp.sw[3] ^ sp.sw[1], 1u);
        // W[20] = rotl(W[17]^W[12]^W[6]^W[4], 1) — all uniform
        sp.sw20 = rotl(sp.sw17 ^ sp.cw12 ^ sp.sw[6] ^ sp.sw[4], 1u);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint32_t lo = uint32_t(salt & 0xFFu);
    uint32_t w10 = 0x20202020u ^ salt_lut[lo >> 4];
    uint32_t w11 = 0x20202020u ^ salt_lut[lo & 0xFu];

    uint32_t a = sp.a, b = sp.b, c = sp.c, d = sp.d, e = sp.e;

    uint32_t w0  = sp.sw[0], w1  = sp.sw[1], w2  = sp.sw[2], w3  = sp.sw[3],
             w4  = sp.sw[4], w5  = sp.sw[5], w6  = sp.sw[6], w7  = sp.sw[7],
             w8  = sp.sw[8], w9  = sp.sw[9];
    uint32_t w12 = sp.cw12, w13 = sp.cw13, w14 = sp.cw14, w15 = sp.cw15;

    // Rounds 10-15 with K fused
    uint32_t wk;
    wk=w10+K0; RK0(wk) wk=w11+K0; RK0(wk) wk=w12+K0; RK0(wk)
    wk=w13+K0; RK0(wk) wk=w14+K0; RK0(wk) wk=w15+K0; RK0(wk)

    // 16-17: precomputed uniform
    uint32_t w16 = sp.sw16; wk=w16+K0; RK0(wk)
    uint32_t w17 = sp.sw17; wk=w17+K0; RK0(wk)

    // 18-79: expand + round with K fused
    uint32_t w18 = rotl(w15 ^ w10 ^ w4  ^ w2,  1u); wk=w18+K0; RK0(wk)
    uint32_t w19 = rotl(w16 ^ w11 ^ w5  ^ w3,  1u); wk=w19+K0; RK0(wk)
    uint32_t w20 = sp.sw20;                                      wk=w20+K1; RK1(wk)
    uint32_t w21 = rotl(w18 ^ w13 ^ w7  ^ w5,  1u); wk=w21+K1; RK1(wk)
    uint32_t w22 = rotl(w19 ^ w14 ^ w8  ^ w6,  1u); wk=w22+K1; RK1(wk)
    uint32_t w23 = rotl(w20 ^ w15 ^ w9  ^ w7,  1u); wk=w23+K1; RK1(wk)
    uint32_t w24 = rotl(w21 ^ w16 ^ w10 ^ w8,  1u); wk=w24+K1; RK1(wk)
    uint32_t w25 = rotl(w22 ^ w17 ^ w11 ^ w9,  1u); wk=w25+K1; RK1(wk)
    uint32_t w26 = rotl(w23 ^ w18 ^ w12 ^ w10, 1u); wk=w26+K1; RK1(wk)
    uint32_t w27 = rotl(w24 ^ w19 ^ w13 ^ w11, 1u); wk=w27+K1; RK1(wk)
    uint32_t w28 = rotl(w25 ^ w20 ^ w14 ^ w12, 1u); wk=w28+K1; RK1(wk)
    uint32_t w29 = rotl(w26 ^ w21 ^ w15 ^ w13, 1u); wk=w29+K1; RK1(wk)
    uint32_t w30 = rotl(w27 ^ w22 ^ w16 ^ w14, 1u); wk=w30+K1; RK1(wk)
    uint32_t w31 = rotl(w28 ^ w23 ^ w17 ^ w15, 1u); wk=w31+K1; RK1(wk)
    uint32_t w32 = rotl(w29 ^ w24 ^ w18 ^ w16, 1u); wk=w32+K1; RK1(wk)
    uint32_t w33 = rotl(w30 ^ w25 ^ w19 ^ w17, 1u); wk=w33+K1; RK1(wk)
    uint32_t w34 = rotl(w31 ^ w26 ^ w20 ^ w18, 1u); wk=w34+K1; RK1(wk)
    uint32_t w35 = rotl(w32 ^ w27 ^ w21 ^ w19, 1u); wk=w35+K1; RK1(wk)
    uint32_t w36 = rotl(w33 ^ w28 ^ w22 ^ w20, 1u); wk=w36+K1; RK1(wk)
    uint32_t w37 = rotl(w34 ^ w29 ^ w23 ^ w21, 1u); wk=w37+K1; RK1(wk)
    uint32_t w38 = rotl(w35 ^ w30 ^ w24 ^ w22, 1u); wk=w38+K1; RK1(wk)
    uint32_t w39 = rotl(w36 ^ w31 ^ w25 ^ w23, 1u); wk=w39+K1; RK1(wk)
    uint32_t w40 = rotl(w37 ^ w32 ^ w26 ^ w24, 1u); wk=w40+K2; RK2(wk)
    uint32_t w41 = rotl(w38 ^ w33 ^ w27 ^ w25, 1u); wk=w41+K2; RK2(wk)
    uint32_t w42 = rotl(w39 ^ w34 ^ w28 ^ w26, 1u); wk=w42+K2; RK2(wk)
    uint32_t w43 = rotl(w40 ^ w35 ^ w29 ^ w27, 1u); wk=w43+K2; RK2(wk)
    uint32_t w44 = rotl(w41 ^ w36 ^ w30 ^ w28, 1u); wk=w44+K2; RK2(wk)
    uint32_t w45 = rotl(w42 ^ w37 ^ w31 ^ w29, 1u); wk=w45+K2; RK2(wk)
    uint32_t w46 = rotl(w43 ^ w38 ^ w32 ^ w30, 1u); wk=w46+K2; RK2(wk)
    uint32_t w47 = rotl(w44 ^ w39 ^ w33 ^ w31, 1u); wk=w47+K2; RK2(wk)
    uint32_t w48 = rotl(w45 ^ w40 ^ w34 ^ w32, 1u); wk=w48+K2; RK2(wk)
    uint32_t w49 = rotl(w46 ^ w41 ^ w35 ^ w33, 1u); wk=w49+K2; RK2(wk)
    uint32_t w50 = rotl(w47 ^ w42 ^ w36 ^ w34, 1u); wk=w50+K2; RK2(wk)
    uint32_t w51 = rotl(w48 ^ w43 ^ w37 ^ w35, 1u); wk=w51+K2; RK2(wk)
    uint32_t w52 = rotl(w49 ^ w44 ^ w38 ^ w36, 1u); wk=w52+K2; RK2(wk)
    uint32_t w53 = rotl(w50 ^ w45 ^ w39 ^ w37, 1u); wk=w53+K2; RK2(wk)
    uint32_t w54 = rotl(w51 ^ w46 ^ w40 ^ w38, 1u); wk=w54+K2; RK2(wk)
    uint32_t w55 = rotl(w52 ^ w47 ^ w41 ^ w39, 1u); wk=w55+K2; RK2(wk)
    uint32_t w56 = rotl(w53 ^ w48 ^ w42 ^ w40, 1u); wk=w56+K2; RK2(wk)
    uint32_t w57 = rotl(w54 ^ w49 ^ w43 ^ w41, 1u); wk=w57+K2; RK2(wk)
    uint32_t w58 = rotl(w55 ^ w50 ^ w44 ^ w42, 1u); wk=w58+K2; RK2(wk)
    uint32_t w59 = rotl(w56 ^ w51 ^ w45 ^ w43, 1u); wk=w59+K2; RK2(wk)
    uint32_t w60 = rotl(w57 ^ w52 ^ w46 ^ w44, 1u); wk=w60+K3; RK3(wk)
    uint32_t w61 = rotl(w58 ^ w53 ^ w47 ^ w45, 1u); wk=w61+K3; RK3(wk)
    uint32_t w62 = rotl(w59 ^ w54 ^ w48 ^ w46, 1u); wk=w62+K3; RK3(wk)
    uint32_t w63 = rotl(w60 ^ w55 ^ w49 ^ w47, 1u); wk=w63+K3; RK3(wk)
    uint32_t w64 = rotl(w61 ^ w56 ^ w50 ^ w48, 1u); wk=w64+K3; RK3(wk)
    uint32_t w65 = rotl(w62 ^ w57 ^ w51 ^ w49, 1u); wk=w65+K3; RK3(wk)
    uint32_t w66 = rotl(w63 ^ w58 ^ w52 ^ w50, 1u); wk=w66+K3; RK3(wk)
    uint32_t w67 = rotl(w64 ^ w59 ^ w53 ^ w51, 1u); wk=w67+K3; RK3(wk)
    uint32_t w68 = rotl(w65 ^ w60 ^ w54 ^ w52, 1u); wk=w68+K3; RK3(wk)
    uint32_t w69 = rotl(w66 ^ w61 ^ w55 ^ w53, 1u); wk=w69+K3; RK3(wk)
    uint32_t w70 = rotl(w67 ^ w62 ^ w56 ^ w54, 1u); wk=w70+K3; RK3(wk)
    uint32_t w71 = rotl(w68 ^ w63 ^ w57 ^ w55, 1u); wk=w71+K3; RK3(wk)
    uint32_t w72 = rotl(w69 ^ w64 ^ w58 ^ w56, 1u); wk=w72+K3; RK3(wk)
    uint32_t w73 = rotl(w70 ^ w65 ^ w59 ^ w57, 1u); wk=w73+K3; RK3(wk)
    uint32_t w74 = rotl(w71 ^ w66 ^ w60 ^ w58, 1u); wk=w74+K3; RK3(wk)
    uint32_t w75 = rotl(w72 ^ w67 ^ w61 ^ w59, 1u); wk=w75+K3; RK3(wk)
    uint32_t w76 = rotl(w73 ^ w68 ^ w62 ^ w60, 1u); wk=w76+K3; RK3(wk)
    uint32_t w77 = rotl(w74 ^ w69 ^ w63 ^ w61, 1u); wk=w77+K3; RK3(wk)
    uint32_t w78 = rotl(w75 ^ w70 ^ w64 ^ w62, 1u); wk=w78+K3; RK3(wk)
    uint32_t w79 = rotl(w76 ^ w71 ^ w65 ^ w63, 1u); wk=w79+K3; RK3(wk)

    uint32_t h0 = sp.h0_init + a;
    if ((h0 & sp.mask0) != sp.target0) return;

    if (sp.prefix_len > 8) {
        uint32_t h1 = sp.h1_init + b;
        if ((h1 & sp.mask1) != sp.target1) return;
    }

    uint expected = 0;
    if (atomic_compare_exchange_weak_explicit(found, &expected, 1u,
            memory_order_relaxed, memory_order_relaxed)) {
        uint32_t h[5] = {h0, sp.h1_init+b, sp.h2_init+c, sp.h3_init+d, sp.h4_init+e};
        device uint8_t* out = result + 4;
        for (int i = 0; i < 8; ++i) out[i] = uint8_t(salt >> (i*8));
        for (int i = 0; i < 5; ++i) {
            result[12+i*4]   = uint8_t(h[i]>>24);
            result[12+i*4+1] = uint8_t(h[i]>>16);
            result[12+i*4+2] = uint8_t(h[i]>>8);
            result[12+i*4+3] = uint8_t(h[i]);
        }
    }
}
