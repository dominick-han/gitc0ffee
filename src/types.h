#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

using HexDigest = std::array<char, 40>;

inline std::string hex_digest_to_string(const HexDigest& h) {
    return {h.data(), h.size()};
}

inline std::vector<uint8_t> hex_string_to_bytes(const std::string& s) {
    auto nib = [](char c) -> uint8_t {
        return (c >= '0' && c <= '9') ? c - '0' :
               (c >= 'a' && c <= 'f') ? c - 'a' + 10 :
               (c >= 'A' && c <= 'F') ? c - 'A' + 10 : 0;
    };
    std::vector<uint8_t> out;
    out.reserve((s.size() + 1) / 2);
    for (size_t i = 0; i + 1 < s.size(); i += 2)
        out.push_back((nib(s[i]) << 4) | nib(s[i + 1]));
    if (s.size() % 2) out.push_back(nib(s.back()) << 4);
    return out;
}

struct ObjectTemplate {
    std::vector<uint8_t> bytes;
    int payload_offset = 0;
    int salt_offset = 0;

    void set_salt(uint64_t salt) {
        // Nibble LUT: each 4-bit value maps to 4 bytes of space/tab encoding.
        // Bit 0 in each nibble position: space (0x20) = 0, tab (0x09) = 1.
        // XOR mask against 0x20202020 (all spaces).
        static constexpr uint32_t lut[16] = {
            0x20202020u, 0x20202009u, 0x20200920u, 0x20200909u,
            0x20092020u, 0x20092009u, 0x20090920u, 0x20090909u,
            0x09202020u, 0x09202009u, 0x09200920u, 0x09200909u,
            0x09092020u, 0x09092009u, 0x09090920u, 0x09090909u,
        };
        uint8_t* dst = bytes.data() + salt_offset;
        // 48 bits = 12 nibbles = 12 words of 4 bytes each
        for (int i = 0; i < 12; ++i) {
            uint32_t w = lut[(salt >> ((11 - i) * 4)) & 0xF];
            dst[i * 4]     = static_cast<uint8_t>(w >> 24);
            dst[i * 4 + 1] = static_cast<uint8_t>(w >> 16);
            dst[i * 4 + 2] = static_cast<uint8_t>(w >> 8);
            dst[i * 4 + 3] = static_cast<uint8_t>(w);
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
