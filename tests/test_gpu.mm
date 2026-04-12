// test_gpu.mm — end-to-end GPU solver tests
//
// Verifies that gpu_solve() finds correct SHA1 prefix matches for
// various prefix lengths and commit shapes. Each test:
//   - Builds a template from a synthetic commit
//   - Runs gpu_solve with a target prefix
//   - Verifies the returned hash starts with the prefix
//   - Verifies the hash matches CPU SHA1 of the salted object

#include "commit.h"
#include "gpu_solver.h"

#include <cstdio>
#include <openssl/sha.h>
#include <string>
#include <vector>

static int g_run = 0, g_pass = 0, g_fail = 0;

static std::string cpu_sha1_hex(const std::vector<uint8_t>& data) {
    unsigned char hash[20];
    SHA_CTX ctx;
    SHA1_Init(&ctx);
    SHA1_Update(&ctx, data.data(), data.size());
    SHA1_Final(hash, &ctx);
    static constexpr char hx[] = "0123456789abcdef";
    std::string out(40, '0');
    for (int i = 0; i < 20; ++i) {
        out[i*2]   = hx[hash[i] >> 4];
        out[i*2+1] = hx[hash[i] & 0xF];
    }
    return out;
}

static std::vector<uint8_t> make_commit(const std::string& msg) {
    std::string s;
    s += "tree 92bf07d030d13a213d73457d73e9070b48caeb58\n";
    s += "parent a461befc8df777d770728186827695919da52feb\n";
    s += "author Test <test@test.com> 1700000000 +0000\n";
    s += "committer Test <test@test.com> 1700000000 +0000\n";
    s += "\n" + msg;
    return {s.begin(), s.end()};
}

static void check_gpu(const std::string& label, const std::string& prefix,
                      const std::string& msg) {
    ++g_run;

    auto raw = make_commit(msg);
    auto obj = parse_commit(raw);
    auto tpl = prepare_template(obj);
    auto result = gpu_solve(tpl, prefix);

    if (!result) {
        fprintf(stderr, "not ok %d - %s: gpu_solve returned nullopt\n", g_run, label.c_str());
        ++g_fail;
        return;
    }

    auto hash = hex_digest_to_string(result->hash);

    // Check prefix match
    bool match = true;
    for (size_t i = 0; i < prefix.size() && i < hash.size(); ++i) {
        if (tolower(prefix[i]) != hash[i]) { match = false; break; }
    }
    if (!match) {
        fprintf(stderr, "not ok %d - %s: hash %s doesn't start with %s\n",
                g_run, label.c_str(), hash.c_str(), prefix.c_str());
        ++g_fail;
        return;
    }

    // Verify hash matches CPU SHA1 of the full object
    // Reconstruct the full git object from the payload
    auto& payload = result->payload;
    std::string hdr = "commit " + std::to_string(payload.size()) + '\0';
    std::vector<uint8_t> full;
    full.insert(full.end(), hdr.begin(), hdr.end());
    full.insert(full.end(), payload.begin(), payload.end());
    auto cpu_hash = cpu_sha1_hex(full);

    if (hash != cpu_hash) {
        fprintf(stderr, "not ok %d - %s: GPU hash %s != CPU hash %s\n",
                g_run, label.c_str(), hash.c_str(), cpu_hash.c_str());
        ++g_fail;
        return;
    }

    fprintf(stderr, "ok %d - %s: %s (prefix=%s)\n",
            g_run, label.c_str(), hash.c_str(), prefix.c_str());
    ++g_pass;
}

int main() {
    fprintf(stderr, "TAP version 13\n");

    // Short prefixes (instant)
    check_gpu("2-nibble prefix", "aa", "short\n");
    check_gpu("3-nibble prefix (odd)", "bad", "odd prefix\n");
    check_gpu("4-nibble prefix", "f00d", "four nibbles\n");
    check_gpu("5-nibble prefix (odd)", "c0ffe", "five nibbles\n");
    check_gpu("6-nibble prefix", "c0ffee", "six nibbles\n");

    // Medium prefix (should take <1s)
    check_gpu("7-nibble prefix (odd)", "deadbee", "seven nibbles\n");

    // 8-nibble prefix (~2s on M4 Pro)
    check_gpu("8-nibble prefix", "c0ffeec0", "eight nibbles\n");

    // Different commit shapes with 6-nibble prefix
    check_gpu("long message", "c0ffee", std::string(500, 'X') + "\n");
    check_gpu("empty message", "c0ffee", "\n");
    check_gpu("multi-line", "c0ffee",
        "feat: something\n\nLong description here.\n\n- item 1\n- item 2\n");

    // Re-run: solve on an already-salted commit
    {
        ++g_run;
        auto raw = make_commit("re-run test\n");
        auto obj = parse_commit(raw);
        auto tpl = prepare_template(obj);
        auto r1 = gpu_solve(tpl, "aa");
        if (!r1) {
            fprintf(stderr, "not ok %d - re-run: first solve failed\n", g_run);
            ++g_fail;
        } else {
            // Feed the salted payload back through
            auto obj2 = parse_commit(r1->payload);
            auto tpl2 = prepare_template(obj2);
            auto r2 = gpu_solve(tpl2, "bb");
            if (!r2) {
                fprintf(stderr, "not ok %d - re-run: second solve failed\n", g_run);
                ++g_fail;
            } else {
                auto h = hex_digest_to_string(r2->hash);
                if (h.substr(0, 2) == "bb") {
                    fprintf(stderr, "ok %d - re-run: %s\n", g_run, h.c_str());
                    ++g_pass;
                } else {
                    fprintf(stderr, "not ok %d - re-run: %s doesn't start with bb\n", g_run, h.c_str());
                    ++g_fail;
                }
            }
        }
    }

    fprintf(stderr, "1..%d\n", g_run);
    if (g_fail > 0)
        fprintf(stderr, "# FAILED %d of %d\n", g_fail, g_run);
    else
        fprintf(stderr, "# All %d tests passed\n", g_pass);

    return g_fail > 0 ? 1 : 0;
}
