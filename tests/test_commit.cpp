// test_commit.cpp - template alignment & SHA1 consistency tests
//
// Verifies that prepare_template() produces correct git object templates
// for a wide variety of commit shapes: short/long authors, merges, GPG
// signatures, unicode, multi-line messages, etc.
//
// Each test checks:
//   - Salt is 4-byte aligned within its SHA1 block
//   - Salt fits within a single 64-byte block
//   - "commit <size>\0" header is consistent
//   - Salt region is zeroed
//   - SHA1 is deterministic across identical salts
//   - Different salts produce different hashes
//   - payload() round-trips through "commit <size>\0" + payload
//   - Original headers are preserved

#include "commit.h"

#include <cstring>
#include <iostream>
#include <openssl/sha.h>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static int g_run = 0, g_pass = 0, g_fail = 0;

#define ASSERT(cond, ...) do { \
    if (!(cond)) { \
        std::cerr << "not ok " << g_run << " - " << label \
                  << ": " << __VA_ARGS__ << "\n"; \
        ++g_fail; return; \
    } \
} while (0)

static std::string cpu_sha1(const std::vector<uint8_t>& data) {
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

static std::vector<uint8_t> make_commit(
    const std::string& tree,
    const std::vector<std::string>& parents,
    const std::string& author,
    const std::string& committer,
    const std::string& message,
    const std::vector<std::string>& extra_headers = {})
{
    std::string s;
    s += "tree " + tree + "\n";
    for (auto& p : parents) s += "parent " + p + "\n";
    s += "author " + author + "\n";
    s += "committer " + committer + "\n";
    for (auto& h : extra_headers) s += h + "\n";
    s += "\n" + message;
    return {s.begin(), s.end()};
}

// ---------------------------------------------------------------------------
// Core check - runs all assertions on a single template
// ---------------------------------------------------------------------------

static void check(const std::string& label, const std::vector<uint8_t>& raw) {
    ++g_run;
    auto obj = parse_commit(raw);
    auto tpl = prepare_template(obj);

    int salt_in_block = tpl.salt_offset % 64;
    ASSERT(salt_in_block % 4 == 0,
           "salt_offset " << tpl.salt_offset << " not 4-byte aligned (block offset " << salt_in_block << ")");
    ASSERT(salt_in_block + 48 <= 64,
           "salt spans block boundary (block offset " << salt_in_block << ")");

    int tail = static_cast<int>(tpl.bytes.size()) - (tpl.salt_offset / 64) * 64;
    ASSERT(tail > 0, "non-positive tail " << tail);

    // "commit <size>\0" header
    std::string expect_hdr = "commit " + std::to_string(tpl.bytes.size() - tpl.payload_offset) + '\0';
    std::string actual_hdr(tpl.bytes.begin(), tpl.bytes.begin() + tpl.payload_offset);
    ASSERT(actual_hdr == expect_hdr, "git object header mismatch");

    // Salt placeholder: 48 spaces (space = bit value 0)
    std::string salt_region(tpl.bytes.begin() + tpl.salt_offset,
                            tpl.bytes.begin() + tpl.salt_offset + 48);
    ASSERT(salt_region == std::string(48, ' '), "salt not zeroed");

    // Determinism
    auto a = tpl; a.set_salt(0xDEADBEEFCAFE);
    auto b = tpl; b.set_salt(0xDEADBEEFCAFE);
    ASSERT(cpu_sha1(a.bytes) == cpu_sha1(b.bytes), "SHA1 not deterministic");

    // Different salt -> different hash
    auto c = tpl; c.set_salt(0x1234567890AB);
    ASSERT(cpu_sha1(a.bytes) != cpu_sha1(c.bytes), "different salts same hash");

    // payload() round-trip
    auto payload = a.payload();
    std::string pfx = "commit " + std::to_string(payload.size()) + '\0';
    std::vector<uint8_t> rebuilt;
    rebuilt.insert(rebuilt.end(), pfx.begin(), pfx.end());
    rebuilt.insert(rebuilt.end(), payload.begin(), payload.end());
    ASSERT(cpu_sha1(a.bytes) == cpu_sha1(rebuilt), "payload() round-trip mismatch");

    // Header preservation
    auto reparsed = parse_commit(payload);
    for (auto& orig : obj.headers) {
        if (orig.starts_with("c0ffeesalt ") || orig.starts_with("c0ffeepad ")) continue;
        bool found = false;
        for (auto& h : reparsed.headers) if (h == orig) { found = true; break; }
        ASSERT(found, "lost header: " << orig);
    }

    std::cerr << "ok " << g_run << " - " << label
              << " (size=" << tpl.bytes.size() << " salt@" << tpl.salt_offset
              << " tail=" << tail << ")\n";
    ++g_pass;
}

// ---------------------------------------------------------------------------
// Test cases
// ---------------------------------------------------------------------------

int main() {
    const std::string tree   = "92bf07d030d13a213d73457d73e9070b48caeb58";
    const std::string parent = "a461befc8df777d770728186827695919da52feb";
    const std::string parent2 = "b572d1c1da28d53f9a6a2201f8a8309fb1097cf6";
    const std::string ts = "1776011275 -0400";

    auto author = [&](const std::string& name, const std::string& email) {
        return name + " <" + email + "> " + ts;
    };

    std::cerr << "TAP version 13\n";

    check("short author",
        make_commit(tree, {parent},
            author("abc", "abc@abc.com"), author("abc", "abc@abc.com"), "GOOD\n"));

    check("typical author",
        make_commit(tree, {parent},
            author("John Doe", "john@example.com"), author("John Doe", "john@example.com"),
            "feat: add new feature\n"));

    check("long author",
        make_commit(tree, {parent},
            author("Bartholomew Aloysius McSomebody III",
                   "bartholomew.aloysius.mcsomebody.iii@very-long-company-name.example.com"),
            author("Bartholomew Aloysius McSomebody III",
                   "bartholomew.aloysius.mcsomebody.iii@very-long-company-name.example.com"),
            "This is a commit with a very long author name and email\n"));

    check("single char author",
        make_commit(tree, {parent}, author("x", "x@x"), author("x", "x@x"), "x\n"));

    check("initial commit (no parent)",
        make_commit(tree, {},
            author("Alice", "alice@example.com"), author("Alice", "alice@example.com"),
            "Initial commit\n"));

    check("merge commit (2 parents)",
        make_commit(tree, {parent, parent2},
            author("Bob", "bob@example.com"), author("Bob", "bob@example.com"),
            "Merge branch 'feature' into main\n"));

    check("octopus merge (3 parents)",
        make_commit(tree, {parent, parent2, "cccccccccccccccccccccccccccccccccccccccc"},
            author("Carol", "carol@example.com"), author("Carol", "carol@example.com"),
            "Octopus merge\n"));

    check("different author/committer",
        make_commit(tree, {parent},
            author("Author Person", "author@example.com"),
            author("Committer Person", "committer@example.com"),
            "Authored by someone, committed by another\n"));

    check("gpg signed commit",
        make_commit(tree, {parent},
            author("Signer", "signer@example.com"), author("Signer", "signer@example.com"),
            "Signed commit\n",
            {"gpgsig -----BEGIN PGP SIGNATURE-----\n"
             " \n"
             " iQEzBAABCAAdFiEE0123456789abcdef0123456789abcdef01234567\n"
             " 89abcdef0123456789abcdef0123456789abcdef0123456789abcdef\n"
             " =ABCD\n"
             " -----END PGP SIGNATURE-----"}));

    check("multi-line message",
        make_commit(tree, {parent},
            author("Dev", "dev@example.com"), author("Dev", "dev@example.com"),
            "feat: big feature\n\nDetailed description.\n\n- Point 1\n- Point 2\n"));

    check("empty message",
        make_commit(tree, {parent},
            author("Dev", "dev@example.com"), author("Dev", "dev@example.com"), "\n"));

    check("unicode author",
        make_commit(tree, {parent},
            "Ñoño García <nono@example.com> " + ts,
            "Ñoño García <nono@example.com> " + ts,
            "Commit with unicode author\n"));

    check("long message (multi-block)",
        make_commit(tree, {parent},
            author("Dev", "dev@example.com"), author("Dev", "dev@example.com"),
            std::string(2000, 'A') + "\n"));

    check("extra headers",
        make_commit(tree, {parent},
            author("Dev", "dev@example.com"), author("Dev", "dev@example.com"),
            "Commit with extra headers\n",
            {"encoding UTF-8", "mergetag object " + parent2}));

    check("minimal commit",
        make_commit(tree, {}, "a <a@a> 0 +0000", "a <a@a> 0 +0000", "\n"));

    // Re-run idempotency: template from a previously salted commit
    {
        auto raw = make_commit(tree, {parent},
            author("Dev", "dev@example.com"), author("Dev", "dev@example.com"),
            "test\n");
        auto obj = parse_commit(raw);
        auto tpl = prepare_template(obj);
        tpl.set_salt(0xABCDEF012345ULL);
        auto payload = tpl.payload();
        // Re-parse the salted payload and prepare again - should strip old salt
        check("re-run idempotency", payload);
    }

    // Message with trailing whitespace (should be cleaned)
    check("trailing whitespace in message",
        make_commit(tree, {parent},
            author("Dev", "dev@example.com"), author("Dev", "dev@example.com"),
            "message with trailing spaces   \n\n\n"));

    // Very short message (single char)
    check("single char message",
        make_commit(tree, {parent},
            author("Dev", "dev@example.com"), author("Dev", "dev@example.com"), "x\n"));

    // Message with tabs and mixed whitespace
    check("tabs in message",
        make_commit(tree, {parent},
            author("Dev", "dev@example.com"), author("Dev", "dev@example.com"),
            "line1\n\tindented\n\t\tdouble\n"));

    // 5 parents (unusual but valid)
    check("5 parents",
        make_commit(tree, {parent, parent2,
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
            "cccccccccccccccccccccccccccccccccccccccc"},
            author("Dev", "dev@example.com"), author("Dev", "dev@example.com"),
            "Five-way merge\n"));

    // Exact block boundary message (stress alignment)
    check("exact 64-byte aligned message",
        make_commit(tree, {parent},
            author("Dev", "dev@example.com"), author("Dev", "dev@example.com"),
            std::string(64, 'B') + "\n"));

    check("exact 128-byte aligned message",
        make_commit(tree, {parent},
            author("Dev", "dev@example.com"), author("Dev", "dev@example.com"),
            std::string(128, 'C') + "\n"));

    // TAP summary
    std::cerr << "1.." << g_run << "\n";
    if (g_fail > 0)
        std::cerr << "# FAILED " << g_fail << " of " << g_run << "\n";
    else
        std::cerr << "# All " << g_pass << " tests passed\n";

    return g_fail > 0 ? 1 : 0;
}
