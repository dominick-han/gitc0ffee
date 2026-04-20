#include "git/git.h"

#include <algorithm>
#include <array>
#include <fcntl.h>
#include <optional>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace git {

// RAII pipe pair. Closes both ends on destruction; take() / close_end() let
// us relinquish ownership piecewise.
struct Pipe {
    int fd[2] = {-1, -1};
    Pipe() { if (pipe(fd) < 0) throw std::runtime_error("pipe failed"); }
    ~Pipe() { for (int& f : fd) if (f >= 0) ::close(f); }
    Pipe(const Pipe&) = delete;
    Pipe& operator=(const Pipe&) = delete;
    void close_end(int i) { if (fd[i] >= 0) { ::close(fd[i]); fd[i] = -1; } }
};

// Fork+exec `argv` (NUL terminator added automatically), optionally piping
// `stdin_data` to its stdin; return its stdout. Stderr is routed to /dev/null.
static std::vector<uint8_t> run_git(std::initializer_list<const char*> argv,
                                    const std::vector<uint8_t>* stdin_data = nullptr) {
    Pipe out;
    std::optional<Pipe> in;
    if (stdin_data) in.emplace();

    const pid_t pid = fork();
    if (pid < 0) throw std::runtime_error("fork failed");

    if (pid == 0) {
        // Child: wire up std{in,out,err}, then exec.
        if (in) { dup2(in->fd[0], STDIN_FILENO); in.reset(); }
        dup2(out.fd[1], STDOUT_FILENO);
        out.close_end(0); out.close_end(1);
        if (int nul = open("/dev/null", O_WRONLY); nul >= 0) {
            dup2(nul, STDERR_FILENO); ::close(nul);
        }
        std::vector<char*> av;
        av.reserve(argv.size() + 1);
        for (const char* s : argv) av.push_back(const_cast<char*>(s));
        av.push_back(nullptr);
        execvp(av[0], av.data());
        _exit(1);
    }

    // Parent: feed stdin (if any), drain stdout, reap.
    if (in) {
        in->close_end(0);
        const uint8_t* p = stdin_data->data();
        for (size_t rem = stdin_data->size(); rem > 0;) {
            const auto n = ::write(in->fd[1], p, rem);
            if (n <= 0) break;
            p += n; rem -= n;
        }
        in.reset();
    }
    out.close_end(1);

    std::vector<uint8_t> result;
    std::array<char, 4096> buf;
    for (;;) {
        const auto n = ::read(out.fd[0], buf.data(), buf.size());
        if (n <= 0) break;
        result.insert(result.end(), buf.data(), buf.data() + n);
    }
    waitpid(pid, nullptr, 0);
    return result;
}

// Copy a 40-char hex digest out of a byte buffer (trailing \n stripped).
static HexDigest to_digest(std::vector<uint8_t> bytes) {
    while (!bytes.empty() && bytes.back() == '\n') bytes.pop_back();
    if (bytes.size() != 40) throw std::runtime_error("unexpected git output");
    HexDigest d;
    std::copy(bytes.begin(), bytes.end(), d.begin());
    return d;
}

HexDigest get_head_digest() {
    return to_digest(run_git({"git", "rev-parse", "HEAD"}));
}

std::vector<uint8_t> get_commit_contents(const HexDigest& digest) {
    const std::string hash(digest.data(), digest.size());
    return run_git({"git", "cat-file", "-p", hash.c_str()});
}

HexDigest write_object(const std::string& type, const std::vector<uint8_t>& contents) {
    return to_digest(run_git(
        {"git", "hash-object", "-w", "-t", type.c_str(), "--stdin"}, &contents));
}

void update_reference(const std::string& ref, const std::string& hash) {
    run_git({"git", "update-ref", ref.c_str(), hash.c_str()});
}

}  // namespace git
