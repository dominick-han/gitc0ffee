#include "git/commit.h"
#include <cstdlib>
#include <iostream>

static constexpr int kSaltLen = 48;

CommitObject parse_commit(const std::vector<uint8_t>& raw) {
    CommitObject obj;
    size_t pos = 0;
    while (pos < raw.size()) {
        size_t nl = pos;
        while (nl < raw.size() && raw[nl] != '\n') ++nl;
        if (nl == pos) { ++pos; break; }
        obj.headers.emplace_back(raw.begin() + pos, raw.begin() + nl);
        pos = nl + 1;
    }
    if (pos < raw.size())
        obj.message.assign(raw.begin() + pos, raw.end());
    return obj;
}

static void clean(CommitObject& obj) {
    std::erase_if(obj.headers, [](const std::string& h) {
        return h.starts_with("c0ffeepad ") || h.starts_with("c0ffeesalt ");
    });
    auto& m = obj.message;
    while (!m.empty() && (m.back() == '\n' || m.back() == ' ' || m.back() == '\t'))
        m.pop_back();
    if (!m.empty()) m.push_back('\n');
}

static ObjectTemplate build_template(const CommitObject& obj, int pad) {
    std::string buf;
    for (auto& h : obj.headers) { buf += h; buf += '\n'; }
    buf += '\n';
    buf += obj.message;
    buf.append(pad, ' ');
    int salt_in_payload = static_cast<int>(buf.size());
    buf.append(kSaltLen, ' ');
    buf += '\n';

    std::string hdr = "commit " + std::to_string(buf.size()) + '\0';
    int po = static_cast<int>(hdr.size());

    std::vector<uint8_t> full;
    full.reserve(hdr.size() + buf.size());
    full.insert(full.end(), hdr.begin(), hdr.end());
    full.insert(full.end(), buf.begin(), buf.end());
    return {std::move(full), po, po + salt_in_payload};
}

// Compute salt_offset and total size for a given pad without allocating.
static std::pair<int,int> offsets(int hdr_bytes, int msg_bytes, int pad) {
    int payload = hdr_bytes + 1 + msg_bytes + pad;  // headers + \n + message + padding
    int salt_in = payload;
    payload += kSaltLen + 1;  // salt + \n

    int digits = 1;
    for (int v = payload; v >= 10; v /= 10) ++digits;
    int po = 7 + digits + 1;  // "commit " + digits + \0
    return {po + salt_in, po + payload};
}

ObjectTemplate prepare_template(const CommitObject& obj) {
    CommitObject c = obj;
    clean(c);

    int hdr = 0;
    for (auto& h : c.headers) hdr += static_cast<int>(h.size()) + 1;
    int msg = static_cast<int>(c.message.size());

    // Find smallest pad where salt is 64-byte aligned and tail fits in 1 block
    for (int pad = 0; pad <= 512; ++pad) {
        auto [so, total] = offsets(hdr, msg, pad);
        if (so % 64 != 0) continue;
        if (total - so <= 55) {
            if (pad > 0)
                std::cerr << "Padding        " << pad << " bytes (1 SHA1 block)\n";
            return build_template(c, pad);
        }
    }

    std::cerr << "Error: could not align salt to 64-byte block boundary\n";
    std::exit(1);
}
