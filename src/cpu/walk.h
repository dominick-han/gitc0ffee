#pragma once
// walk.h - portable salt-enumeration skeleton shared by the non-x86 backends.
//
// The salt's 12 nibbles map one-to-one onto W[0..11] of the SHA-1 message
// schedule. Enumerating salts in order keeps the high nibbles fixed for long
// runs, so the SHA-1 prefix is computed once per level and reused:
//
//   Level 3 (per 65536 salts)  W[0..7]  constant -> rounds 0-7
//   Level 2 (per  4096 salts)  + W[8]   constant -> round  8
//   Level 1 (per   256 salts)  + W[9]   constant -> round  9
//   Level 0 (per    16 salts)  + W[10]  constant -> round  10
//                                W[11]  differs per salt (backend's job)
//
// walk_salts() drives that loop nest and hands each 16-salt block to a
// backend kernel together with the state after round 10.

#include "cpu/common.h"

#include <algorithm>
#include <atomic>

static inline uint32_t sha1_ch(uint32_t b, uint32_t c, uint32_t d) { return (b & c) | (~b & d); }

// One SHA-1 round using the Ch function and K0 (rounds 0-19).
static inline void sha1_round_ch(uint32_t& a, uint32_t& b, uint32_t& c, uint32_t& d, uint32_t& e, uint32_t w) {
    const uint32_t t = rotl32(a, 5) + sha1_ch(b, c, d) + e + SHA1_K[0] + w;
    e = d; d = c; c = rotl32(b, 30); b = a; a = t;
}

// SHA-1 working state (a..e) after some number of rounds.
struct Sha1State { uint32_t a, b, c, d, e; };

// Shared view of one 16-salt block: first salt (16-aligned), state after
// rounds 0..10, and message words W[0..10] (W[11] = salt_lut[salt & 0xF]).
struct SaltBlock {
    uint64_t  salt;
    Sha1State st;
    uint32_t  w[11];
};

// Check a candidate's finalized h0/h1 against the prefix. On a hit, recompute
// the full digest with the reference implementation (so backends never need
// to finalize e), publish it, and return true.
static inline bool check_candidate(const CPUParams& p, uint32_t h0, uint32_t h1, uint64_t salt,
                                   uint64_t salt_start, std::atomic<bool>& global_found,
                                   WorkerResult& result) {
    if (__builtin_expect((h0 & p.mask0) != p.target0, 1)) return false;
    if (p.prefix_len > 8 && (h1 & p.mask1) != p.target1) return false;
    sha1_final_block(p, salt, result.hash);
    result.salt   = salt;
    result.found  = true;
    result.hashes = salt - salt_start + 1;
    global_found.store(true, std::memory_order_relaxed);
    return true;
}

// Enumerate [salt_start, salt_end) in 16-salt blocks. `kernel(const SaltBlock&)`
// hashes all 16 salts of a block and returns true once it has stored a hit.
template <class Kernel>
static inline void walk_salts(const CPUParams& p, uint64_t salt_start, uint64_t salt_end,
                              std::atomic<bool>& global_found, WorkerResult& result,
                              Kernel&& kernel) {
    result.found  = false;
    result.hashes = 0;

    // Truncate to a 16-multiple so no partial blocks escape the loop.
    // (At most 15 salts at the very end of the 48-bit space are skipped.)
    salt_end &= ~uint64_t(15);
    if (salt_start >= salt_end) return;

    SaltBlock blk;
    for (uint64_t salt = salt_start; salt < salt_end; ) {
        Sha1State l3{p.pre_state[0], p.pre_state[1], p.pre_state[2], p.pre_state[3], p.pre_state[4]};
        for (int i = 0; i < 8; ++i) {
            blk.w[i] = salt_lut[(salt >> (44 - 4 * i)) & 0xF];
            sha1_round_ch(l3.a, l3.b, l3.c, l3.d, l3.e, blk.w[i]);
        }
        const uint64_t l3_end = std::min<uint64_t>(salt_end, (salt | 0xFFFFull) + 1);
        if (global_found.load(std::memory_order_relaxed)) return;

        for (uint64_t s3 = salt; s3 < l3_end; s3 = (s3 | 0xFFFull) + 1) {
            blk.w[8] = salt_lut[(s3 >> 12) & 0xF];
            Sha1State l2 = l3;
            sha1_round_ch(l2.a, l2.b, l2.c, l2.d, l2.e, blk.w[8]);
            const uint64_t l2_end = std::min<uint64_t>(l3_end, (s3 | 0xFFFull) + 1);

            for (uint64_t s2 = s3; s2 < l2_end; s2 = (s2 | 0xFFull) + 1) {
                blk.w[9] = salt_lut[(s2 >> 8) & 0xF];
                Sha1State l1 = l2;
                sha1_round_ch(l1.a, l1.b, l1.c, l1.d, l1.e, blk.w[9]);
                const uint64_t l1_end = std::min<uint64_t>(l2_end, (s2 | 0xFFull) + 1);

                for (uint64_t s = s2; s < l1_end; s += 16) {
                    blk.w[10] = salt_lut[(s >> 4) & 0xF];
                    blk.st = l1;
                    sha1_round_ch(blk.st.a, blk.st.b, blk.st.c, blk.st.d, blk.st.e, blk.w[10]);
                    blk.salt = s;
                    if (kernel(blk)) return;
                }
            }
        }
        result.hashes = l3_end - salt_start;
        salt = l3_end;
    }
    result.hashes = salt_end - salt_start;
}
