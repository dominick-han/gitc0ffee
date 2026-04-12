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
        uint8_t* dst = bytes.data() + salt_offset;
        for (int i = 47; i >= 0; --i) {
            dst[i] = (salt & 1) ? '\t' : ' ';
            salt >>= 1;
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
