#include "git/commit.h"
#include "git/git.h"
#include "solver.h"

#include <cstdio>
#include <cstring>
#include <getopt.h>

#ifndef VERSION
#define VERSION "dev"
#endif

static void usage(const char* prog) {
    std::fprintf(stderr,
        "gitc0ffee -- brute-force git commit hash prefixes\n\n"
        "Usage: %s [OPTIONS] [HEX]\n\n"
        "  HEX             hex prefix to match (default: c0ffee)\n"
        "  -w, --write     write object and update HEAD\n"
        "  -b, --backend   force backend (avx512, sha-ni, avx2, metal)\n"
        "  -q, --quiet     suppress progress output\n"
        "  -V, --version   print version\n"
        "  -h, --help      show this help\n", prog);
}

int main(int argc, char** argv) {
    std::string prefix_hex = "c0ffee";
    std::string backend;
    bool update_ref = false;
    bool quiet      = false;

    static const option long_opts[] = {
        {"write",   no_argument,       nullptr, 'w'},
        {"backend", required_argument, nullptr, 'b'},
        {"quiet",   no_argument,       nullptr, 'q'},
        {"version", no_argument,       nullptr, 'V'},
        {"help",    no_argument,       nullptr, 'h'},
        {}
    };

    for (int c; (c = getopt_long(argc, argv, "wb:qVh", long_opts, nullptr)) != -1;) {
        switch (c) {
            case 'w': update_ref = true; break;
            case 'b': backend    = optarg; break;
            case 'q': quiet      = true; break;
            case 'V': std::fprintf(stderr, "gitc0ffee %s\n", VERSION); return 0;
            case 'h': usage(argv[0]); return 0;
            default:  usage(argv[0]); return 1;
        }
    }
    if (optind < argc) prefix_hex = argv[optind];

    // Validate: at most 40 hex chars.
    if (prefix_hex.size() > 40) {
        std::fprintf(stderr, "Error: prefix can't exceed 40 hex chars (SHA1 is 20 bytes)\n");
        return 1;
    }
    if (std::strspn(prefix_hex.c_str(), "0123456789abcdefABCDEF") != prefix_hex.size()) {
        std::fprintf(stderr, "Error: prefix must be a hex string (got '%s')\n", prefix_hex.c_str());
        return 1;
    }

    if (quiet) std::freopen("/dev/null", "w", stderr);

    std::fprintf(stderr, "Target Prefix  %s (%zu nibble%s)\n",
                 prefix_hex.c_str(), prefix_hex.size(), prefix_hex.size() == 1 ? "" : "s");

    auto tpl = prepare_template(parse_commit(git::get_commit_contents(git::get_head_digest())));
    std::fprintf(stderr, "Object Size    %zu bytes\nSalt Offset    %d\n\n",
                 tpl.bytes.size(), tpl.salt_offset);

    auto result = solve(tpl, prefix_hex, backend);
    if (!result) { std::printf("No Solution Found\n"); return 1; }

    const auto hash    = hex_digest_to_string(result->hash);
    const auto written = git::write_object("commit", result->payload);
    if (hex_digest_to_string(written) != hash) {
        std::printf("Hash Mismatch!\n");
        return 1;
    }
    if (update_ref) {
        git::update_reference("HEAD", hash);
        std::fprintf(stderr, "HEAD Updated to ");
    }
    std::printf("%s\n", hash.c_str());
    return 0;
}
