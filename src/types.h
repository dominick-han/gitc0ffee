#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using HexDigest = std::array<char, 40>;

inline std::string hex_digest_to_string(const HexDigest& h) {
    return {h.data(), h.size()};
}

// Decode a hex string into bytes. An odd trailing nibble is placed in the
// high half of a final byte. Invalid chars decode as 0 — callers must
// pre-validate.
inline std::vector<uint8_t> hex_string_to_bytes(const std::string& s) {
    static constexpr auto kNibble = [] {
        std::array<uint8_t, 256> t{};
        for (int c = '0'; c <= '9'; ++c) t[c] = c - '0';
        for (int c = 'a'; c <= 'f'; ++c) t[c] = c - 'a' + 10;
        for (int c = 'A'; c <= 'F'; ++c) t[c] = c - 'A' + 10;
        return t;
    }();
    std::vector<uint8_t> out((s.size() + 1) / 2);
    for (size_t i = 0; i < s.size(); ++i) {
        const uint8_t n = kNibble[(uint8_t)s[i]];
        out[i / 2] |= (i & 1) ? n : (n << 4);
    }
    return out;
}

struct ObjectTemplate {
    std::vector<uint8_t> bytes;
    int payload_offset = 0;
    int salt_offset = 0;

    // Stamp a 48-bit salt as 48 bytes of space/tab encoding at salt_offset.
    // Each nibble -> one big-endian 32-bit word: bit k picks space (0x20)
    // vs tab (0x09) at byte-position k (LSB-first within the word).
    void set_salt(uint64_t salt) {
        static constexpr uint32_t kNibbleWord[16] = {
            0x20202020u, 0x20202009u, 0x20200920u, 0x20200909u,
            0x20092020u, 0x20092009u, 0x20090920u, 0x20090909u,
            0x09202020u, 0x09202009u, 0x09200920u, 0x09200909u,
            0x09092020u, 0x09092009u, 0x09090920u, 0x09090909u,
        };
        uint8_t* dst = bytes.data() + salt_offset;
        for (int i = 0; i < 12; ++i) {
            const uint32_t w = __builtin_bswap32(kNibbleWord[(salt >> ((11 - i) * 4)) & 0xF]);
            std::memcpy(dst + i * 4, &w, 4);
        }
    }

    std::vector<uint8_t> payload() const {
        return {bytes.begin() + payload_offset, bytes.end()};
    }
};

struct SolveResult {
    std::vector<uint8_t> payload;
    HexDigest hash;
};
