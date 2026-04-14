#include <metal_stdlib>
using namespace metal;

static constant uint32_t K0 = 0x5A827999u;
static constant uint32_t K1 = 0x6ED9EBA1u;
static constant uint32_t K2 = 0x8F1BBCDCu;
static constant uint32_t K3 = 0xCA62C1D6u;

inline uint32_t rotl(uint32_t x, uint32_t n) {
    return (x << n) | (x >> (32u - n));
}

// Salt encoding: 48-bit salt -> 48 bytes -> 12 words.
// Each byte encodes 1 bit: space (0x20) = 0, tab (0x09) = 1.
// LUT maps each nibble to XOR mask against 0x20202020. Single lookup per word.
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
    uint32_t prefix_len;     // nibble count
    uint32_t _pad[3];
    uint64_t base_salt;
    uint32_t pre_state[5];
    uint32_t block_w[16];    // final SHA1 block (salt words zeroed)
    uint32_t prefix_words[5];
};

#define R0(i) { uint32_t t = rotl(a,5u) + ((b&c)|((~b)&d)) + e + K0 + w##i; e=d; d=c; c=rotl(b,30u); b=a; a=t; }
#define R1(i) { uint32_t t = rotl(a,5u) + (b^c^d)           + e + K1 + w##i; e=d; d=c; c=rotl(b,30u); b=a; a=t; }
#define R2(i) { uint32_t t = rotl(a,5u) + ((b&c)|(b&d)|(c&d)) + e + K2 + w##i; e=d; d=c; c=rotl(b,30u); b=a; a=t; }
#define R3(i) { uint32_t t = rotl(a,5u) + (b^c^d)           + e + K3 + w##i; e=d; d=c; c=rotl(b,30u); b=a; a=t; }

kernel void bruteforce_sha1(
    device const GPUParams& params [[buffer(0)]],
    device uint8_t*         result [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    device auto* found = reinterpret_cast<device atomic_uint*>(result);
    if (atomic_load_explicit(found, memory_order_relaxed) != 0) return;

    uint64_t salt = params.base_salt + uint64_t(tid);

    // Salt occupies w[0..11] (48 bytes, 48 bits), template has w[12..15]
    uint32_t w0  = encode_salt_word(salt, 0);
    uint32_t w1  = encode_salt_word(salt, 1);
    uint32_t w2  = encode_salt_word(salt, 2);
    uint32_t w3  = encode_salt_word(salt, 3);
    uint32_t w4  = encode_salt_word(salt, 4);
    uint32_t w5  = encode_salt_word(salt, 5);
    uint32_t w6  = encode_salt_word(salt, 6);
    uint32_t w7  = encode_salt_word(salt, 7);
    uint32_t w8  = encode_salt_word(salt, 8);
    uint32_t w9  = encode_salt_word(salt, 9);
    uint32_t w10 = encode_salt_word(salt, 10);
    uint32_t w11 = encode_salt_word(salt, 11);
    uint32_t w12 = params.block_w[12];
    uint32_t w13 = params.block_w[13];
    uint32_t w14 = params.block_w[14];
    uint32_t w15 = params.block_w[15];

    uint32_t a = params.pre_state[0], b = params.pre_state[1],
             c = params.pre_state[2], d = params.pre_state[3],
             e = params.pre_state[4];

    // Rounds 0-15
    R0(0)  R0(1)  R0(2)  R0(3)  R0(4)  R0(5)  R0(6)  R0(7)
    R0(8)  R0(9)  R0(10) R0(11) R0(12) R0(13) R0(14) R0(15)

    // Rounds 16-79: fused expand + rounds
    { uint32_t w16 = rotl(w13 ^ w8  ^ w2  ^ w0,  1u); R0(16)
      uint32_t w17 = rotl(w14 ^ w9  ^ w3  ^ w1,  1u); R0(17)
      uint32_t w18 = rotl(w15 ^ w10 ^ w4  ^ w2,  1u); R0(18)
      uint32_t w19 = rotl(w16 ^ w11 ^ w5  ^ w3,  1u); R0(19)
      uint32_t w20 = rotl(w17 ^ w12 ^ w6  ^ w4,  1u); R1(20)
      uint32_t w21 = rotl(w18 ^ w13 ^ w7  ^ w5,  1u); R1(21)
      uint32_t w22 = rotl(w19 ^ w14 ^ w8  ^ w6,  1u); R1(22)
      uint32_t w23 = rotl(w20 ^ w15 ^ w9  ^ w7,  1u); R1(23)
      uint32_t w24 = rotl(w21 ^ w16 ^ w10 ^ w8,  1u); R1(24)
      uint32_t w25 = rotl(w22 ^ w17 ^ w11 ^ w9,  1u); R1(25)
      uint32_t w26 = rotl(w23 ^ w18 ^ w12 ^ w10, 1u); R1(26)
      uint32_t w27 = rotl(w24 ^ w19 ^ w13 ^ w11, 1u); R1(27)
      uint32_t w28 = rotl(w25 ^ w20 ^ w14 ^ w12, 1u); R1(28)
      uint32_t w29 = rotl(w26 ^ w21 ^ w15 ^ w13, 1u); R1(29)
      uint32_t w30 = rotl(w27 ^ w22 ^ w16 ^ w14, 1u); R1(30)
      uint32_t w31 = rotl(w28 ^ w23 ^ w17 ^ w15, 1u); R1(31)
      uint32_t w32 = rotl(w29 ^ w24 ^ w18 ^ w16, 1u); R1(32)
      uint32_t w33 = rotl(w30 ^ w25 ^ w19 ^ w17, 1u); R1(33)
      uint32_t w34 = rotl(w31 ^ w26 ^ w20 ^ w18, 1u); R1(34)
      uint32_t w35 = rotl(w32 ^ w27 ^ w21 ^ w19, 1u); R1(35)
      uint32_t w36 = rotl(w33 ^ w28 ^ w22 ^ w20, 1u); R1(36)
      uint32_t w37 = rotl(w34 ^ w29 ^ w23 ^ w21, 1u); R1(37)
      uint32_t w38 = rotl(w35 ^ w30 ^ w24 ^ w22, 1u); R1(38)
      uint32_t w39 = rotl(w36 ^ w31 ^ w25 ^ w23, 1u); R1(39)
      uint32_t w40 = rotl(w37 ^ w32 ^ w26 ^ w24, 1u); R2(40)
      uint32_t w41 = rotl(w38 ^ w33 ^ w27 ^ w25, 1u); R2(41)
      uint32_t w42 = rotl(w39 ^ w34 ^ w28 ^ w26, 1u); R2(42)
      uint32_t w43 = rotl(w40 ^ w35 ^ w29 ^ w27, 1u); R2(43)
      uint32_t w44 = rotl(w41 ^ w36 ^ w30 ^ w28, 1u); R2(44)
      uint32_t w45 = rotl(w42 ^ w37 ^ w31 ^ w29, 1u); R2(45)
      uint32_t w46 = rotl(w43 ^ w38 ^ w32 ^ w30, 1u); R2(46)
      uint32_t w47 = rotl(w44 ^ w39 ^ w33 ^ w31, 1u); R2(47)
      uint32_t w48 = rotl(w45 ^ w40 ^ w34 ^ w32, 1u); R2(48)
      uint32_t w49 = rotl(w46 ^ w41 ^ w35 ^ w33, 1u); R2(49)
      uint32_t w50 = rotl(w47 ^ w42 ^ w36 ^ w34, 1u); R2(50)
      uint32_t w51 = rotl(w48 ^ w43 ^ w37 ^ w35, 1u); R2(51)
      uint32_t w52 = rotl(w49 ^ w44 ^ w38 ^ w36, 1u); R2(52)
      uint32_t w53 = rotl(w50 ^ w45 ^ w39 ^ w37, 1u); R2(53)
      uint32_t w54 = rotl(w51 ^ w46 ^ w40 ^ w38, 1u); R2(54)
      uint32_t w55 = rotl(w52 ^ w47 ^ w41 ^ w39, 1u); R2(55)
      uint32_t w56 = rotl(w53 ^ w48 ^ w42 ^ w40, 1u); R2(56)
      uint32_t w57 = rotl(w54 ^ w49 ^ w43 ^ w41, 1u); R2(57)
      uint32_t w58 = rotl(w55 ^ w50 ^ w44 ^ w42, 1u); R2(58)
      uint32_t w59 = rotl(w56 ^ w51 ^ w45 ^ w43, 1u); R2(59)
      uint32_t w60 = rotl(w57 ^ w52 ^ w46 ^ w44, 1u); R3(60)
      uint32_t w61 = rotl(w58 ^ w53 ^ w47 ^ w45, 1u); R3(61)
      uint32_t w62 = rotl(w59 ^ w54 ^ w48 ^ w46, 1u); R3(62)
      uint32_t w63 = rotl(w60 ^ w55 ^ w49 ^ w47, 1u); R3(63)
      uint32_t w64 = rotl(w61 ^ w56 ^ w50 ^ w48, 1u); R3(64)
      uint32_t w65 = rotl(w62 ^ w57 ^ w51 ^ w49, 1u); R3(65)
      uint32_t w66 = rotl(w63 ^ w58 ^ w52 ^ w50, 1u); R3(66)
      uint32_t w67 = rotl(w64 ^ w59 ^ w53 ^ w51, 1u); R3(67)
      uint32_t w68 = rotl(w65 ^ w60 ^ w54 ^ w52, 1u); R3(68)
      uint32_t w69 = rotl(w66 ^ w61 ^ w55 ^ w53, 1u); R3(69)
      uint32_t w70 = rotl(w67 ^ w62 ^ w56 ^ w54, 1u); R3(70)
      uint32_t w71 = rotl(w68 ^ w63 ^ w57 ^ w55, 1u); R3(71)
      uint32_t w72 = rotl(w69 ^ w64 ^ w58 ^ w56, 1u); R3(72)
      uint32_t w73 = rotl(w70 ^ w65 ^ w59 ^ w57, 1u); R3(73)
      uint32_t w74 = rotl(w71 ^ w66 ^ w60 ^ w58, 1u); R3(74)
      uint32_t w75 = rotl(w72 ^ w67 ^ w61 ^ w59, 1u); R3(75)
      uint32_t w76 = rotl(w73 ^ w68 ^ w62 ^ w60, 1u); R3(76)
      uint32_t w77 = rotl(w74 ^ w69 ^ w63 ^ w61, 1u); R3(77)
      uint32_t w78 = rotl(w75 ^ w70 ^ w64 ^ w62, 1u); R3(78)
      uint32_t w79 = rotl(w76 ^ w71 ^ w65 ^ w63, 1u); R3(79)
    }

    uint32_t h0 = params.pre_state[0] + a;

    // Fast prefix check on h0 (covers <=8 nibble prefixes without computing h1-h4)
    uint32_t pn = params.prefix_len;
    if (pn <= 8) {
        uint32_t mask = (pn == 8) ? 0xFFFFFFFFu : (0xFFFFFFFFu << ((8u - pn) * 4u));
        if ((h0 & mask) != (params.prefix_words[0] & mask)) return;
    } else {
        if (h0 != params.prefix_words[0]) return;
        uint32_t h1 = params.pre_state[1] + b;
        uint32_t rem = pn - 8;
        uint32_t mask = (rem >= 8) ? 0xFFFFFFFFu : (0xFFFFFFFFu << ((8u - rem) * 4u));
        if ((h1 & mask) != (params.prefix_words[1] & mask)) return;
        if (rem > 8) {
            uint32_t h2 = params.pre_state[2] + c; rem -= 8;
            mask = (rem >= 8) ? 0xFFFFFFFFu : (0xFFFFFFFFu << ((8u - rem) * 4u));
            if ((h2 & mask) != (params.prefix_words[2] & mask)) return;
        }
    }

    // Match - write result
    uint expected = 0;
    if (atomic_compare_exchange_weak_explicit(found, &expected, 1u,
            memory_order_relaxed, memory_order_relaxed)) {
        uint32_t h[5] = {h0, params.pre_state[1]+b, params.pre_state[2]+c,
                          params.pre_state[3]+d, params.pre_state[4]+e};
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
