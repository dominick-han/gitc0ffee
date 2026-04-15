#include "git/commit.h"
#include "git/git.h"
#include "solver.h"

#include <getopt.h>
#include <iostream>

#ifndef VERSION
#define VERSION "dev"
#endif

static void usage(const char* prog) {
    std::cerr << "gitc0ffee -- brute-force git commit hash prefixes\n\n"
              << "Usage: " << prog << " [OPTIONS] [HEX]\n\n"
              << "  HEX             hex prefix to match (default: c0ffee)\n"
              << "  -w, --write     write object and update HEAD\n"
              << "  -b, --backend   force backend (avx512, sha-ni, avx2, metal)\n"
              << "  -q, --quiet     suppress progress output\n"
              << "  -V, --version   print version\n"
              << "  -h, --help      show this help\n";
}

int main(int argc, char** argv) {
    std::string prefix_hex = "c0ffee";
    std::string backend;
    bool update_ref = false;
    bool quiet = false;

    static const struct option long_opts[] = {
        {"write",   no_argument,       nullptr, 'w'},
        {"backend", required_argument, nullptr, 'b'},
        {"quiet",   no_argument,       nullptr, 'q'},
        {"version", no_argument,       nullptr, 'V'},
        {"help",    no_argument,       nullptr, 'h'},
        {nullptr, 0, nullptr, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "wb:qVh", long_opts, nullptr)) != -1) {
        switch (c) {
            case 'w': update_ref = true; break;
            case 'b': backend = optarg; break;
            case 'q': quiet = true; break;
            case 'V': std::cerr << "gitc0ffee " << VERSION << "\n"; return 0;
            case 'h': usage(argv[0]); return 0;
            default:  usage(argv[0]); return 1;
        }
    }

    if (optind < argc) prefix_hex = argv[optind];

    if (prefix_hex.size() > 40) {
        std::cerr << "Error: prefix can't exceed 40 hex chars (SHA1 is 20 bytes)\n"; return 1;
    }
    for (char ch : prefix_hex) {
        if (!((ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f') || (ch >= 'A' && ch <= 'F'))) {
            std::cerr << "Error: prefix must be a hex string (got '" << ch << "')\n"; return 1;
        }
    }

    if (quiet) std::freopen("/dev/null", "w", stderr);

    std::cerr << "Target Prefix  " << prefix_hex
              << " (" << prefix_hex.size() << " nibble" << (prefix_hex.size() != 1 ? "s" : "") << ")\n";

    auto head = git::get_head_digest();
    auto tpl = prepare_template(parse_commit(git::get_commit_contents(head)));

    std::cerr << "Object Size    " << tpl.bytes.size() << " bytes\n"
              << "Salt Offset    " << tpl.salt_offset << "\n\n";

    auto result = solve(tpl, prefix_hex, backend);
    if (!result) { std::cout << "No Solution Found\n"; return 1; }

    auto hash = hex_digest_to_string(result->hash);
    auto written = git::write_object("commit", result->payload);
    if (hex_digest_to_string(written) != hash) {
        std::cout << "Hash Mismatch!\n"; return 1;
    }

    if (update_ref) {
        git::update_reference("HEAD", hash);
        std::cerr << "HEAD Updated to ";
    }

    std::cout << hash << "\n";
    return 0;
}
