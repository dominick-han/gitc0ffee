#include "commit.h"
#include "git.h"
#include "solver.h"

#include <iostream>

#ifndef VERSION
#define VERSION "dev"
#endif

int main(int argc, char** argv) {
    std::string prefix_hex = "c0ffee";
    bool update_ref = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--prefix" && i + 1 < argc) prefix_hex = argv[++i];
        else if (arg == "--update-ref") update_ref = true;
        else if (arg == "-V" || arg == "--version") {
            std::cerr << "gitc0ffee " << VERSION << "\n"; return 0;
        } else if (arg == "-h" || arg == "--help") {
            std::cerr << "gitc0ffee - brute-force git commit hash prefixes\n\n"
                      << "Usage: " << argv[0] << " [--prefix HEX] [--update-ref]\n\n"
                      << "  --prefix HEX    hex prefix to match (default: c0ffee)\n"
                      << "  --update-ref    rewrite HEAD to the new commit\n"
                      << "  -V, --version   print version\n"
                      << "  -h, --help      show this help\n";
            return 0;
        } else {
            std::cerr << "Unknown option: " << arg << "\n"; return 1;
        }
    }

    if (prefix_hex.size() > 40) {
        std::cerr << "Error: prefix can't exceed 20 bytes (40 hex chars) - SHA1 is 20 bytes\n"; return 1;
    }
    for (char c : prefix_hex) {
        if (!((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'))) {
            std::cerr << "Error: prefix must be a hex string (got '" << c << "')\n"; return 1;
        }
    }

    std::cerr << "Target Prefix  " << prefix_hex
              << " (" << prefix_hex.size() << " nibble" << (prefix_hex.size() != 1 ? "s" : "") << ")\n";

    auto head = git::get_head_digest();
    auto tpl = prepare_template(parse_commit(git::get_commit_contents(head)));

    std::cerr << "Object Size    " << tpl.bytes.size() << " bytes\n"
              << "Salt Offset    " << tpl.salt_offset << "\n\n";

    auto result = solve(tpl, prefix_hex);
    if (!result) { std::cerr << "No Solution Found\n"; return 1; }

    auto hash = hex_digest_to_string(result->hash);
    auto written = git::write_object("commit", result->payload);
    if (hex_digest_to_string(written) != hash) {
        std::cerr << "Hash Mismatch!\n"; return 1;
    }

    if (update_ref) {
        git::update_reference("HEAD", hash);
        std::cerr << "HEAD Updated!\n";
    }

    std::cout << hash << "\n";
    return 0;
}
