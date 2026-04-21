// GPU SHA-1 brute-force kernel for git vanity commit hashes.
//
// Each thread hashes one candidate salt. The 48-byte salt encodes 48 bits of
// entropy as a run of spaces and tabs, filling W[0..11] of the final SHA-1
// block; W[12..15] is the uniform padding tail.
//
// Salts within a 256-thread threadgroup share bits 8..47, so W[0..9] is
// TG-uniform; bits 4..7 (W[10]'s nibble) give 16 groups of 16 threads; bits
// 0..3 (W[11]'s nibble) are per-thread. Sixteen lanes precompute rounds 0..10
// into threadgroup memory (one per W[10] nibble) and round 11 collapses to
// one add per thread: after round 10 the state is nibble-uniform, so
// t = rotl(a,5) + f(...) + e + K + W[11] becomes (per_nibble_scalar + W[11]).

#include <metal_stdlib>
using namespace metal;

static constant uint32_t K0 = 0x5A827999u, K1 = 0x6ED9EBA1u,
                         K2 = 0x8F1BBCDCu, K3 = 0xCA62C1D6u;

// Nibble -> 4-byte space/tab word (bit k picks tab at byte k, LSB-first).
static constant uint32_t salt_lut[16] = {
    0x20202020u, 0x20202009u, 0x20200920u, 0x20200909u,
    0x20092020u, 0x20092009u, 0x20090920u, 0x20090909u,
    0x09202020u, 0x09202009u, 0x09200920u, 0x09200909u,
    0x09092020u, 0x09092009u, 0x09090920u, 0x09090909u,
};

inline uint32_t rotl (uint32_t x, uint32_t n) { return (x << n) | (x >> (32u - n)); }
inline uint32_t f_ch (uint32_t b, uint32_t c, uint32_t d) { return (b & c) ^ (~b & d); }
inline uint32_t f_par(uint32_t b, uint32_t c, uint32_t d) { return b ^ c ^ d; }
inline uint32_t f_maj(uint32_t b, uint32_t c, uint32_t d) { return (b & c) ^ (b & d) ^ (c & d); }

struct GPUParams {
    uint32_t mask0, target0, mask1, target1;
    uint32_t prefix_len;
    uint32_t pre_state[5];
    uint32_t tail_w[4];         // W[12..15] of the final block
    uint32_t _pad;
    uint64_t base_salt;
};

// One SHA-1 round, with K folded into wk to save one add on the critical path.
#define ROUND(f, wk) do { \
    uint32_t t = rotl(a, 5u) + f(b, c, d) + e + (wk); \
    e = d; d = c; c = rotl(b, 30u); b = a; a = t; \
} while (0)

// W[i] = rotl(W[i-3] ^ W[i-8] ^ W[i-14] ^ W[i-16], 1), via rolling 16-word buffer.
#define EXPAND(i) (w[(i)&15] = rotl(w[((i)-3)&15] ^ w[((i)-8)&15] \
                                  ^ w[((i)-14)&15] ^ w[((i)-16)&15], 1u))

kernel void bruteforce_sha1(
    constant GPUParams& params [[buffer(0)]],
    device uint8_t*     result [[buffer(1)]],
    uint tid [[thread_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    device auto* found = reinterpret_cast<device atomic_uint*>(result);
    if (atomic_load_explicit(found, memory_order_relaxed)) return;

    uint64_t salt = params.base_salt + tid;

    // Threadgroup precompute, stages merged into one pass to drop a barrier:
    // lanes 0-15 of simdgroup 0 each run rounds 0-10 for their W[10] nibble.
    // Rounds 0-9 are TG-uniform so the 16 lanes do them redundantly in SIMD
    // lockstep (~10 cycles total, not 10 * 16). Round 10 diverges per nibble.
    // The post-round-11-collapse state goes into the threadgroup cache, keyed
    // by nibble. Also cache W[0..9] for the main body.
    threadgroup uint32_t sh_sw[10];
    threadgroup uint32_t sh_t11[16], sh_b[16], sh_c[16], sh_d[16], sh_e[16];

    if (lid < 16) {
        uint32_t sw[10];
        for (int i = 0; i < 10; ++i) sw[i] = salt_lut[(salt >> ((11 - i) * 4)) & 0xFu];
        if (lid == 0) for (int i = 0; i < 10; ++i) sh_sw[i] = sw[i];

        uint32_t a = params.pre_state[0], b = params.pre_state[1],
                 c = params.pre_state[2], d = params.pre_state[3],
                 e = params.pre_state[4];
        for (int i = 0; i < 10; ++i) ROUND(f_ch, sw[i] + K0);
        ROUND(f_ch, salt_lut[lid] + K0);    // round 10 with this nibble's W[10]

        sh_t11[lid] = rotl(a, 5u) + f_ch(b, c, d) + e + K0;
        sh_b[lid] = a; sh_c[lid] = rotl(b, 30u); sh_d[lid] = c; sh_e[lid] = d;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Main body. Round 11 collapses to one add: only `a` is per-thread.
    uint32_t nib10 = (salt >> 4) & 0xFu, nib11 = salt & 0xFu;
    uint32_t a = sh_t11[nib10] + salt_lut[nib11];
    uint32_t b = sh_b[nib10], c = sh_c[nib10], d = sh_d[nib10], e = sh_e[nib10];

    uint32_t w[16];
    for (int i = 0; i < 10; ++i) w[i] = sh_sw[i];
    w[10] = salt_lut[nib10]; w[11] = salt_lut[nib11];
    w[12] = params.tail_w[0]; w[13] = params.tail_w[1];
    w[14] = params.tail_w[2]; w[15] = params.tail_w[3];

    ROUND(f_ch, w[12] + K0); ROUND(f_ch, w[13] + K0);
    ROUND(f_ch, w[14] + K0); ROUND(f_ch, w[15] + K0);
    for (int i = 16; i < 20; ++i) { EXPAND(i); ROUND(f_ch,  w[i&15] + K0); }
    for (int i = 20; i < 40; ++i) { EXPAND(i); ROUND(f_par, w[i&15] + K1); }
    for (int i = 40; i < 60; ++i) { EXPAND(i); ROUND(f_maj, w[i&15] + K2); }
    for (int i = 60; i < 80; ++i) { EXPAND(i); ROUND(f_par, w[i&15] + K3); }

    // Prefix check on the finalized hash (h0 rejects most candidates).
    uint32_t h0 = params.pre_state[0] + a;
    if ((h0 & params.mask0) != params.target0) return;
    if (params.prefix_len > 8) {
        uint32_t h1 = params.pre_state[1] + b;
        if ((h1 & params.mask1) != params.target1) return;
    }

    // Claim the result slot; first writer wins.
    uint expected = 0;
    if (!atomic_compare_exchange_weak_explicit(found, &expected, 1u,
            memory_order_relaxed, memory_order_relaxed)) return;

    uint32_t h[5] = { h0, params.pre_state[1] + b, params.pre_state[2] + c,
                          params.pre_state[3] + d, params.pre_state[4] + e };
    for (int i = 0; i < 8; ++i) result[4 + i] = uint8_t(salt >> (i * 8));
    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 4; ++j)
            result[12 + i*4 + j] = uint8_t(h[i] >> ((3 - j) * 8));
}
