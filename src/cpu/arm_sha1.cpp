// arm_sha1.cpp - ARMv8 Crypto Extensions backend: 4-way interleaved SHA-1.
//
// Neoverse-class cores execute every SHA1 instruction (sha1c/p/m, and also
// sha1su0/su1/h) on ONE pipe: 1 op/cycle, 4-cycle latency (measured on
// Graviton3). Rounds 12-79 are 17 sha1c/p/m quads, so the floor is ~17
// cycles/hash and everything else has to stay off that pipe:
//
//   * Rounds 0-10 come from the shared level precompute in walk.h; round 11
//     (the first per-salt word) is one scalar add folded into A.
//   * The message schedule is GF(2)-linear in the block words. Salts in a
//     16-salt block differ only in W[11], so
//         W_salt[t] = W_base[t] ^ Delta11[nibble][t]
//     with Delta11 a compile-time table. W_base is expanded once per 256
//     salts (W[10] = W[11] = 0) and patched per block with a Delta10 table.
//     Per salt, the schedule costs one eor + one add (K) per quad on the
//     plain ASIMD pipes, with no sha1su0/sha1su1 in the hot loop.
//   * E for the next quad, rol30(A), is a lane move plus a scalar ror on an
//     integer pipe. sha1h would take a slot on the SHA1 pipe.
//   * 4 lanes covers the 4-cycle latency. Wider interleaving spills: each
//     lane's eor/add chain needs the 17 W_base quads live too (measured 4.95
//     GH/s at 4 lanes vs 4.22 at 8 and 2.90 at 16 on 64 cores).
//   * W+K is computed inline. Staging it in a per-block table costs one more
//     store + load per quad and measured ~35% slower.
// Only h0 (and h1 for >8-nibble prefixes) is finalized in the hot loop; full
// digests are recomputed by the reference on a hit.

#if !defined(__aarch64__)
#error "arm_sha1.cpp requires aarch64"
#endif

// Enable the SHA1 instructions for this TU only; dispatch is guarded at
// runtime by HWCAP_SHA1, so the rest of the binary stays baseline ARMv8.
#pragma GCC target("+crypto")

#include "cpu/walk.h"

#include <arm_neon.h>
#include <array>
#include <utility>

namespace {

constexpr int kLanes = 4;
static_assert(16 % kLanes == 0, "lanes must divide the 16-salt block");

// Schedule words 12..79 as 17 quads (quad index Q-3 for rounds 4Q..4Q+3).
constexpr int kQuads = 17;
using Sched = std::array<uint32_t, kQuads * 4>;

// Expand a 16-word block to words 12..79.
constexpr Sched expand(const std::array<uint32_t, 16>& in) {
    std::array<uint32_t, 80> w{};
    for (int i = 0; i < 16; ++i) w[i] = in[i];
    for (int i = 16; i < 80; ++i) {
        const uint32_t x = w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16];
        w[i] = (x << 1) | (x >> 31);
    }
    Sched out{};
    for (int i = 0; i < kQuads * 4; ++i) out[i] = w[12 + i];
    return out;
}

// Delta[n] = schedule of a block that is all-zero except word `Pos` = salt_lut[n].
template <int Pos>
constexpr std::array<Sched, 16> make_delta() {
    std::array<Sched, 16> d{};
    for (int n = 0; n < 16; ++n) {
        std::array<uint32_t, 16> in{};
        in[Pos] = salt_lut[n];
        d[n] = expand(in);
    }
    return d;
}
alignas(64) constexpr std::array<Sched, 16> kDelta11 = make_delta<11>();
alignas(64) constexpr std::array<Sched, 16> kDelta10 = make_delta<10>();

// Quads whose W[11] delta is zero for every nibble need no per-salt eor.
constexpr std::array<bool, kQuads> make_delta_live() {
    std::array<bool, kQuads> live{};
    for (int q = 0; q < kQuads; ++q)
        for (int n = 0; n < 16; ++n)
            for (int j = 0; j < 4; ++j)
                if (kDelta11[n][q * 4 + j]) live[q] = true;
    return live;
}
constexpr std::array<bool, kQuads> kDeltaLive = make_delta_live();

// One quad (rounds 4Q..4Q+3, Q in [3, 19]) across all lanes; lane i hashes
// the salt whose low nibble is n0 + i.
template <int Q>
__attribute__((always_inline)) inline void quad(uint32x4_t (&abcd)[kLanes], uint32_t (&e)[kLanes],
                                                const uint32x4_t* base, int n0) {
    constexpr int F  = Q / 5;   // 0=Ch, 1=Parity, 2=Majority, 3=Parity
    constexpr int QI = Q - 3;
    #pragma GCC unroll 4
    for (int i = 0; i < kLanes; ++i) {
        const uint32_t ne = rotl32(vgetq_lane_u32(abcd[i], 0), 30);   // next quad's E
        uint32x4_t w = base[QI];
        if constexpr (kDeltaLive[QI]) w = veorq_u32(w, vld1q_u32(kDelta11[n0 + i].data() + QI * 4));
        w = vaddq_u32(w, vdupq_n_u32(SHA1_K[F]));
        if constexpr (F == 0)      abcd[i] = vsha1cq_u32(abcd[i], e[i], w);
        else if constexpr (F == 2) abcd[i] = vsha1mq_u32(abcd[i], e[i], w);
        else                       abcd[i] = vsha1pq_u32(abcd[i], e[i], w);
        e[i] = ne;
    }
}

template <int... Q>
__attribute__((always_inline)) inline void rounds_12_79(uint32x4_t (&abcd)[kLanes], uint32_t (&e)[kLanes],
                                                        const uint32x4_t* base, int n0,
                                                        std::integer_sequence<int, Q...>) {
    (quad<Q + 3>(abcd, e, base, n0), ...);
}

} // namespace

void arm_sha1_worker(const CPUParams& p, uint64_t salt_start, uint64_t salt_end,
                     std::atomic<bool>& global_found, WorkerResult& result) {
    alignas(16) uint32x4_t wl1[kQuads];     // schedule with W[10] = W[11] = 0, per 256 salts
    alignas(16) uint32x4_t wbase[kQuads];   // schedule with W[11] = 0, per 16 salts
    uint64_t cur_l1 = ~uint64_t(0);

    walk_salts(p, salt_start, salt_end, global_found, result, [&](const SaltBlock& blk) {
        // Level 1 (W[0..9] fixed): full scalar expansion, amortized over 256 salts.
        if ((blk.salt >> 8) != cur_l1) {
            cur_l1 = blk.salt >> 8;
            std::array<uint32_t, 16> in{};
            for (int i = 0; i < 10; ++i) in[i] = blk.w[i];
            for (int i = 12; i < 16; ++i) in[i] = p.msg_words[i];
            const Sched s = expand(in);
            for (int q = 0; q < kQuads; ++q) wl1[q] = vld1q_u32(&s[q * 4]);
        }
        // Level 0 (+ W[10]): patch with the W[10] delta.
        const uint32_t* d10 = kDelta10[(blk.salt >> 4) & 0xF].data();
        for (int q = 0; q < kQuads; ++q) wbase[q] = veorq_u32(wl1[q], vld1q_u32(d10 + q * 4));

        // Round 11: B/C/D/E of the round-12 state don't depend on W[11], so
        // fold everything else into A12_BASE; per-salt A12 is one add.
        const Sha1State& s = blk.st;
        const uint32_t A12_BASE = rotl32(s.a, 5) + sha1_ch(s.b, s.c, s.d) + s.e + SHA1_K[0];
        const uint32_t B12 = s.a, C12 = rotl32(s.b, 30), D12 = s.c, E12 = s.d;

        for (int q = 0; q < 16; q += kLanes) {
            uint32x4_t abcd[kLanes];
            uint32_t   e[kLanes];
            #pragma GCC unroll 4
            for (int i = 0; i < kLanes; ++i) {
                const uint32_t st[4] = {A12_BASE + salt_lut[q + i], B12, C12, D12};
                abcd[i] = vld1q_u32(st);
                e[i]    = E12;
            }

            rounds_12_79(abcd, e, wbase, q, std::make_integer_sequence<int, kQuads>{});

            for (int i = 0; i < kLanes; ++i) {
                const uint32_t h0 = p.pre_state[0] + vgetq_lane_u32(abcd[i], 0);
                const uint32_t h1 = p.pre_state[1] + vgetq_lane_u32(abcd[i], 1);
                if (check_candidate(p, h0, h1, blk.salt + q + i, salt_start, global_found, result))
                    return true;
            }
        }
        return false;
    });
}
