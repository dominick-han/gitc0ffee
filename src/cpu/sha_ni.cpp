// sha_ni.cpp - SHA-NI backend: 4-way interleaved SHA-1 brute force
//
// Each inner iteration hashes 4 salts through the SHA-NI pipeline at once
// (interleaved to hide sha1rnds4 latency). The salt's 12 nibbles map one-to-one
// onto W[0..11] of the SHA-1 message schedule. When enumerating salts in
// order, the high nibbles stay fixed for long runs so we precompute the
// SHA-1 prefix once per level and reuse it across many hashes:
//
//   Level 3 (per 65536 salts)  W[0..7]  constant -> precompute rounds 0-7
//   Level 2 (per  4096 salts)  + W[8]   constant -> precompute round  8
//   Level 1 (per   256 salts)  + W[9]   constant -> precompute round  9
//   Level 0 (per    16 salts)  + W[10]  constant -> precompute round 10
//                                W[11]  differs per salt -> fold into A_post11
//
// SHA-NI's sha1rnds4 runs in 4-round blocks, so we can't intervene mid-block.
// We precompute the scalar SHA-1 prefix through round 11 per salt, then hand
// SHA-NI two synthetic xmm registers at round 12 entry:
//   E1 = A_post7   (level-3 invariant; sha1nexte's rol+add produces E^12)
//   A  = A_post11  (per salt; A field = A12_BASE + W[11])
// The msg1/xor operations that R4_7/R8_11 would normally do to M0, M1 are
// replayed as two precomputed broadcasts so rounds 12-79 run unmodified.

#include "cpu/common.h"
#include <algorithm>
#include <atomic>
#include <immintrin.h>

// Scalar SHA-1 round, used for the lane-uniform precompute phase.
static inline uint32_t f_ch_s(uint32_t b, uint32_t c, uint32_t d) { return (b & c) | (~b & d); }
static inline void scalar_round(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, uint32_t& e, uint32_t w, uint32_t k) {
    uint32_t t = rotl32(a, 5) + f_ch_s(b, c, d) + e + k + w;
    e = d; d = c; c = rotl32(b, 30); b = a; a = t;
}

// Pack scalar SHA-1 state (a,b,c,d) into a SHA-NI ABCD register.
// SHA-NI layout: xmm[127:96]=A, xmm[95:64]=B, xmm[63:32]=C, xmm[31:0]=D.
static inline __m128i pack_abcd(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
    return _mm_shuffle_epi32(_mm_setr_epi32((int)a, (int)b, (int)c, (int)d), 0x1B);
}

// -------- SHA-NI body-round macros ---------------------------------------
// Each ROUND_Mk macro drives one 4-round block where Mk is the "driver"
// message quad feeding sha1rnds4 / sha1nexte. The four macros correspond
// to the four positions in the message-schedule ring (M0 -> M1 -> M2 -> M3
// -> M0 ...). Per-lane state lives in arrays: M0[i]..M3[i] (message
// schedule quads), A[i] (ABCD state), E[i]/E1[i] (ping-pong E-carry
// registers). `fn` selects the round function (0=Ch, 1/3=Parity, 2=Majority).
#define ROUND_M0(i,fn) E[i]=_mm_sha1nexte_epu32(E[i],M0[i]);  E1[i]=A[i]; M1[i]=_mm_sha1msg2_epu32(M1[i],M0[i]); A[i]=_mm_sha1rnds4_epu32(A[i],E[i],fn);  M3[i]=_mm_sha1msg1_epu32(M3[i],M0[i]); M2[i]=_mm_xor_si128(M2[i],M0[i]);
#define ROUND_M1(i,fn) E1[i]=_mm_sha1nexte_epu32(E1[i],M1[i]); E[i]=A[i]; M2[i]=_mm_sha1msg2_epu32(M2[i],M1[i]); A[i]=_mm_sha1rnds4_epu32(A[i],E1[i],fn); M0[i]=_mm_sha1msg1_epu32(M0[i],M1[i]); M3[i]=_mm_xor_si128(M3[i],M1[i]);
#define ROUND_M2(i,fn) E[i]=_mm_sha1nexte_epu32(E[i],M2[i]);  E1[i]=A[i]; M3[i]=_mm_sha1msg2_epu32(M3[i],M2[i]); A[i]=_mm_sha1rnds4_epu32(A[i],E[i],fn);  M1[i]=_mm_sha1msg1_epu32(M1[i],M2[i]); M0[i]=_mm_xor_si128(M0[i],M2[i]);
#define ROUND_M3(i,fn) E1[i]=_mm_sha1nexte_epu32(E1[i],M3[i]); E[i]=A[i]; M0[i]=_mm_sha1msg2_epu32(M0[i],M3[i]); A[i]=_mm_sha1rnds4_epu32(A[i],E1[i],fn); M2[i]=_mm_sha1msg1_epu32(M2[i],M3[i]); M1[i]=_mm_xor_si128(M1[i],M3[i]);

// Compare one salt's finalized hash against the prefix target. On a hit, write
// the full digest into `result` and return true.
__attribute__((target("sha,sse4.1,ssse3")))
static inline bool check_and_store(__m128i A, __m128i E_before_fin, __m128i E0_INIT, const CPUParams& p, uint64_t salt, uint64_t salt_start, std::atomic<bool>& global_found, WorkerResult& result) {
    __m128i ABCD_out = _mm_shuffle_epi32(A, 0x1B);
    uint32_t h0 = (uint32_t)_mm_extract_epi32(ABCD_out, 0);
    if (__builtin_expect((h0 & p.mask0) != p.target0, 1)) return false;
    if (p.prefix_len > 8 && ((uint32_t)_mm_extract_epi32(ABCD_out, 1) & p.mask1) != p.target1) return false;

    __m128i E_final = _mm_sha1nexte_epu32(E_before_fin, E0_INIT);
    result.salt = salt;
    _mm_storeu_si128((__m128i*)result.hash, ABCD_out);
    result.hash[4] = (uint32_t)_mm_extract_epi32(E_final, 3);
    result.found = true;
    result.hashes = salt - salt_start + 1;
    global_found.store(true, std::memory_order_relaxed);
    return true;
}

__attribute__((target("sha,sse4.1,ssse3")))
void sha_ni_worker(const CPUParams& p, uint64_t salt_start, uint64_t salt_end, std::atomic<bool>& global_found, WorkerResult& result) {
    result.found  = false;
    result.hashes = 0;

    // The 4-way batch steps by 4 salts at a time and the level-0 loop by 16.
    // Truncate salt_end to a 16-multiple so no partial batches escape the loop.
    // (At most 15 salts at the very end of the 48-bit space are skipped.)
    salt_end &= ~uint64_t(15);
    if (salt_start >= salt_end) return;

    const __m128i ABCD_INIT = pack_abcd(p.pre_state[0], p.pre_state[1], p.pre_state[2], p.pre_state[3]);
    const __m128i E0_INIT   = _mm_set_epi32((int)p.pre_state[4], 0, 0, 0);
    const __m128i MSG3_C    = p.msg3_const;

    // Level 3: per 65536 salts. W[0..7] constant -> scalar rounds 0-7.
    for (uint64_t salt = salt_start; salt < salt_end; ) {
        uint32_t sw[11];
        uint32_t l3a = p.pre_state[0], l3b = p.pre_state[1], l3c = p.pre_state[2], l3d = p.pre_state[3], l3e = p.pre_state[4];
        for (int i = 0; i < 8; ++i) {
            sw[i] = salt_lut[(salt >> (44 - 4*i)) & 0xF];
            scalar_round(l3a, l3b, l3c, l3d, l3e, sw[i], SHA1_K[0]);
        }

        // M0, M1 hold pristine W[0..3], W[4..7]. M0_AFTER_R4_7 is the sha1msg1 that
        // R4_7 would do to M0 (level-3 invariant: depends only on W[0..7]). ABCD_POST7
        // is ABCD_post7, which becomes the E1 carry at round 12 entry.
        const __m128i M0_L3         = _mm_set_epi32((int)sw[0], (int)sw[1], (int)sw[2], (int)sw[3]);
        const __m128i M1_L3         = _mm_set_epi32((int)sw[4], (int)sw[5], (int)sw[6], (int)sw[7]);
        const __m128i M0_AFTER_R4_7 = _mm_sha1msg1_epu32(M0_L3, M1_L3);
        const __m128i ABCD_POST7    = pack_abcd(l3a, l3b, l3c, l3d);

        const uint64_t l3_end = std::min<uint64_t>(salt_end, (salt | 0xFFFFull) + 1);
        if (global_found.load(std::memory_order_relaxed)) return;

        // Level 2: per 4096 salts. + W[8] constant -> scalar round 8.
        for (uint64_t s3 = salt; s3 < l3_end; s3 = (s3 | 0xFFFull) + 1) {
            sw[8] = salt_lut[(s3 >> 12) & 0xF];
            uint32_t l2a = l3a, l2b = l3b, l2c = l3c, l2d = l3d, l2e = l3e;
            scalar_round(l2a, l2b, l2c, l2d, l2e, sw[8], SHA1_K[0]);
            const uint64_t l2_end = std::min<uint64_t>(l3_end, (s3 | 0xFFFull) + 1);

            // Level 1: per 256 salts. + W[9] constant -> scalar round 9.
            for (uint64_t s2 = s3; s2 < l2_end; s2 = (s2 | 0xFFull) + 1) {
                sw[9] = salt_lut[(s2 >> 8) & 0xF];
                uint32_t l1a = l2a, l1b = l2b, l1c = l2c, l1d = l2d, l1e = l2e;
                scalar_round(l1a, l1b, l1c, l1d, l1e, sw[9], SHA1_K[0]);

                // sha1msg1(M1, M2) only reads M1 (W[4..7]) and M2's top half (W[8..9]),
                // so M1's post-R8_11 state is a level-1 invariant.
                const __m128i M1_AFTER_R8_11 = _mm_sha1msg1_epu32(M1_L3, _mm_set_epi32((int)sw[8], (int)sw[9], 0, 0));
                const uint64_t l1_end = std::min<uint64_t>(l2_end, (s2 | 0xFFull) + 1);

                // Level 0: 16 salts at a time. + W[10] constant -> scalar round 10.
                for (uint64_t s = s2; s < l1_end; s += 16) {
                    sw[10] = salt_lut[(s >> 4) & 0xF];
                    uint32_t l0a = l1a, l0b = l1b, l0c = l1c, l0d = l1d, l0e = l1e;
                    scalar_round(l0a, l0b, l0c, l0d, l0e, sw[10], SHA1_K[0]);

                    // Scalar round 11 is per-salt. B12/C12/D12 don't depend on W[11], so
                    // fold everything except W[11] into A12_BASE; the per-salt A12 is one add.
                    //   A12 = rol(l0a,5) + f_ch(l0b,l0c,l0d) + l0e + K0 + W[11]
                    const uint32_t A12_BASE = rotl32(l0a, 5) + f_ch_s(l0b, l0c, l0d) + l0e + SHA1_K[0];
                    const uint32_t B12 = l0a;
                    const uint32_t C12 = rotl32(l0b, 30);
                    const uint32_t D12 = l0c;

                    // Hash the 16 salts in 4 interleaved batches of 4.
                    for (int q = 0; q < 16; q += 4) {
                        uint32_t w11[4];
                        __m128i A[4], E[4], E1[4], M0[4], M1[4], M2[4], M3[4];
                        for (int i = 0; i < 4; ++i) {
                            w11[i] = salt_lut[(s + q + i) & 0xF];
                            // M2 layout: [127:96]=W[8] [95:64]=W[9] [63:32]=W[10] [31:0]=W[11]
                            M2[i] = _mm_set_epi32((int)sw[8], (int)sw[9], (int)sw[10], (int)w11[i]);
                            // After R8_11, M0 = sha1msg1(M0,M1) XOR M2. The sha1msg1 piece
                            // is level-3 invariant (M0_AFTER_R4_7); XOR with per-salt M2.
                            M0[i] = _mm_xor_si128(M0_AFTER_R4_7, M2[i]);
                            M1[i] = M1_AFTER_R8_11;
                            M3[i] = MSG3_C;
                            // Per-salt A12 state: only the A field varies per salt.
                            A[i]  = pack_abcd(A12_BASE + w11[i], B12, C12, D12);
                            E[i]  = _mm_setzero_si128();
                            E1[i] = ABCD_POST7;
                        }

                        // -------- SHA-NI rounds 12..79, 4 lanes interleaved --------
                        // 17 body blocks cycling through the 4 driver registers. Rounds
                        // 12-15 is just ROUND_M3(.,0): E1 carries the preloaded A_post7
                        // so sha1nexte's rol+add produces E^12 on the fly. In the tail
                        // (rounds 64+), the dead schedule ops (msg1/msg2/xor whose
                        // results are never read before finalization) are DCE'd by the
                        // compiler since M1/M2/M3 are not consumed past round 79.
                        #define PARALLEL(ROUND, fn) ROUND(0,fn) ROUND(1,fn) ROUND(2,fn) ROUND(3,fn)
                        PARALLEL(ROUND_M3, 0)  // rounds 12-15
                        PARALLEL(ROUND_M0, 0)  // rounds 16-19
                        PARALLEL(ROUND_M1, 1)  // rounds 20-23
                        PARALLEL(ROUND_M2, 1)  // rounds 24-27
                        PARALLEL(ROUND_M3, 1)  // rounds 28-31
                        PARALLEL(ROUND_M0, 1)  // rounds 32-35
                        PARALLEL(ROUND_M1, 1)  // rounds 36-39
                        PARALLEL(ROUND_M2, 2)  // rounds 40-43
                        PARALLEL(ROUND_M3, 2)  // rounds 44-47
                        PARALLEL(ROUND_M0, 2)  // rounds 48-51
                        PARALLEL(ROUND_M1, 2)  // rounds 52-55
                        PARALLEL(ROUND_M2, 2)  // rounds 56-59
                        PARALLEL(ROUND_M3, 3)  // rounds 60-63
                        PARALLEL(ROUND_M0, 3)  // rounds 64-67
                        PARALLEL(ROUND_M1, 3)  // rounds 68-71
                        PARALLEL(ROUND_M2, 3)  // rounds 72-75
                        PARALLEL(ROUND_M3, 3)  // rounds 76-79
                        #undef PARALLEL

                        const uint64_t base = s + q;
                        for (int i = 0; i < 4; ++i) {
                            __m128i A_final = _mm_add_epi32(A[i], ABCD_INIT);
                            if (check_and_store(A_final, E[i], E0_INIT, p, base + i, salt_start, global_found, result)) return;
                        }
                    }
                }
            }
        }
        result.hashes = l3_end - salt_start;
        salt = l3_end;
    }
    result.hashes = salt_end - salt_start;
}
