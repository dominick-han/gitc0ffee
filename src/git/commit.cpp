#include "git/commit.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static constexpr int kSaltLen = 48;

CommitObject parse_commit(const std::vector<uint8_t>& raw) {
    CommitObject obj;
    size_t pos = 0;
    while (pos < raw.size()) {
        size_t nl = pos;
        while (nl < raw.size() && raw[nl] != '\n') ++nl;
        if (nl == pos) { ++pos; break; }   // empty line terminates headers
        obj.headers.emplace_back(raw.begin() + pos, raw.begin() + nl);
        pos = nl + 1;
    }
    if (pos < raw.size())
        obj.message.assign(raw.begin() + pos, raw.end());
    return obj;
}

// Strip any prior gitc0ffee-authored headers and canonicalize trailing
// whitespace on the commit message (exactly one \n if non-empty).
static void clean(CommitObject& obj) {
    std::erase_if(obj.headers, [](const std::string& h) {
        return h.starts_with("c0ffeepad ") || h.starts_with("c0ffeesalt ");
    });
    auto& m = obj.message;
    while (!m.empty() && (m.back() == '\n' || m.back() == ' ' || m.back() == '\t'))
        m.pop_back();
    if (!m.empty()) m.push_back('\n');
}

// For a given payload size, return the length of the "commit N\0" header.
static int header_len(int payload_bytes) {
    // "commit " + decimal digits + "\0"
    return 7 + (int)std::to_string(payload_bytes).size() + 1;
}

ObjectTemplate prepare_template(const CommitObject& obj_in) {
    CommitObject obj = obj_in;
    clean(obj);

    int hdr_bytes = 0;
    for (auto& h : obj.headers) hdr_bytes += (int)h.size() + 1;  // each line + \n
    const int msg_bytes = (int)obj.message.size();

    // Find the smallest `pad` such that the salt is 64-byte aligned in the
    // full object. With a fixed-size 48-byte salt + trailing "\n", the post-
    // salt tail is always 49 bytes, which fits inside the final SHA-1 block.
    for (int pad = 0; pad <= 512; ++pad) {
        const int headers_plus_msg = hdr_bytes + 1 + msg_bytes + pad;  // + blank line between headers & msg
        const int salt_in_payload  = headers_plus_msg;                 // salt sits here
        const int payload_bytes    = headers_plus_msg + kSaltLen + 1;  // + salt + trailing \n
        const int salt_in_object   = header_len(payload_bytes) + salt_in_payload;

        if (salt_in_object % 64 != 0) continue;

        // Fits: materialize the object.
        std::string payload;
        payload.reserve(payload_bytes);
        for (auto& h : obj.headers) { payload += h; payload += '\n'; }
        payload += '\n';
        payload += obj.message;
        payload.append(pad, ' ');
        payload.append(kSaltLen, ' ');
        payload += '\n';

        const std::string hdr = "commit " + std::to_string(payload.size()) + '\0';
        ObjectTemplate tpl;
        tpl.bytes.reserve(hdr.size() + payload.size());
        tpl.bytes.insert(tpl.bytes.end(), hdr.begin(), hdr.end());
        tpl.bytes.insert(tpl.bytes.end(), payload.begin(), payload.end());
        tpl.payload_offset = (int)hdr.size();
        tpl.salt_offset    = (int)hdr.size() + salt_in_payload;
        if (pad > 0) std::fprintf(stderr, "Padding        %d bytes (1 SHA1 block)\n", pad);
        return tpl;
    }

    std::fprintf(stderr, "Error: could not align salt to 64-byte block boundary\n");
    std::exit(1);
}
