// scalar.cpp - portable scalar SHA-1 backend + reference final-block compressor.
//
// The scalar worker is the fallback on CPUs without SHA extensions (and is
// selectable everywhere with --backend scalar). sha1_final_block() is the
// reference used by the walk.h backends to produce the full digest on a hit.

#include "cpu/walk.h"

// Message word i of the final block for `salt`.
static inline uint32_t block_word(const CPUParams& p, uint64_t salt, int i) {
    return i < 12 ? salt_lut[(salt >> (44 - 4 * i)) & 0xF] : p.msg_words[i];
}

void sha1_final_block(const CPUParams& p, uint64_t salt, uint32_t out[5]) {
    uint32_t w[80];
    for (int i = 0; i < 16; ++i) w[i] = block_word(p, salt, i);
    for (int i = 16; i < 80; ++i) w[i] = rotl32(w[i-3] ^ w[i-8] ^ w[i-14] ^ w[i-16], 1);

    uint32_t a = p.pre_state[0], b = p.pre_state[1], c = p.pre_state[2],
             d = p.pre_state[3], e = p.pre_state[4];
    for (int i = 0; i < 80; ++i) {
        uint32_t f;
        if      (i < 20) f = sha1_ch(b, c, d);
        else if (i < 40) f = b ^ c ^ d;
        else if (i < 60) f = (b & c) | (b & d) | (c & d);
        else             f = b ^ c ^ d;
        const uint32_t t = rotl32(a, 5) + f + e + SHA1_K[i / 20] + w[i];
        e = d; d = c; c = rotl32(b, 30); b = a; a = t;
    }
    out[0] = p.pre_state[0] + a;
    out[1] = p.pre_state[1] + b;
    out[2] = p.pre_state[2] + c;
    out[3] = p.pre_state[3] + d;
    out[4] = p.pre_state[4] + e;
}

void scalar_worker(const CPUParams& p, uint64_t salt_start, uint64_t salt_end,
                   std::atomic<bool>& global_found, WorkerResult& result) {
    walk_salts(p, salt_start, salt_end, global_found, result, [&](const SaltBlock& blk) {
        for (int k = 0; k < 16; ++k) {
            uint32_t w[16];
            for (int i = 0; i < 11; ++i) w[i] = blk.w[i];
            w[11] = salt_lut[k];
            for (int i = 12; i < 16; ++i) w[i] = p.msg_words[i];

            uint32_t a = blk.st.a, b = blk.st.b, c = blk.st.c, d = blk.st.d, e = blk.st.e;
            #pragma GCC unroll 80
            for (int i = 11; i < 80; ++i) {
                uint32_t wi;
                if (i < 16) {
                    wi = w[i];
                } else {
                    wi = rotl32(w[(i-3) & 15] ^ w[(i-8) & 15] ^ w[(i-14) & 15] ^ w[i & 15], 1);
                    w[i & 15] = wi;
                }
                uint32_t f;
                if      (i < 20) f = sha1_ch(b, c, d);
                else if (i < 40) f = b ^ c ^ d;
                else if (i < 60) f = (b & c) | (b & d) | (c & d);
                else             f = b ^ c ^ d;
                const uint32_t t = rotl32(a, 5) + f + e + SHA1_K[i / 20] + wi;
                e = d; d = c; c = rotl32(b, 30); b = a; a = t;
            }
            if (check_candidate(p, p.pre_state[0] + a, p.pre_state[1] + b, blk.salt + k,
                                salt_start, global_found, result))
                return true;
        }
        return false;
    });
}
