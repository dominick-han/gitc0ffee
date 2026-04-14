#include "git.h"

#include <array>
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace git {

// Fork+exec a git command, capture stdout into a byte vector.
// Avoids popen/shell overhead entirely.
static std::vector<uint8_t> exec_git(const std::vector<const char*>& argv) {
    int pipefd[2];
    if (pipe(pipefd) < 0) throw std::runtime_error("pipe failed");

    pid_t pid = fork();
    if (pid < 0) { close(pipefd[0]); close(pipefd[1]); throw std::runtime_error("fork failed"); }

    if (pid == 0) {
        close(pipefd[0]);
        dup2(pipefd[1], STDOUT_FILENO);
        close(pipefd[1]);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
        execvp("git", const_cast<char* const*>(argv.data()));
        _exit(1);
    }

    close(pipefd[1]);
    std::vector<uint8_t> out;
    std::array<char, 4096> buf;
    for (;;) {
        auto n = ::read(pipefd[0], buf.data(), buf.size());
        if (n <= 0) break;
        out.insert(out.end(), buf.data(), buf.data() + n);
    }
    close(pipefd[0]);
    waitpid(pid, nullptr, 0);
    return out;
}

HexDigest get_head_digest() {
    auto out = exec_git({"git", "rev-parse", "HEAD", nullptr});
    // Strip trailing newline
    while (!out.empty() && out.back() == '\n') out.pop_back();
    if (out.size() != 40) throw std::runtime_error("bad HEAD");
    HexDigest d;
    std::copy(out.begin(), out.end(), d.begin());
    return d;
}

std::vector<uint8_t> get_commit_contents(const HexDigest& digest) {
    // Use a stack buffer for the digest string to avoid allocation
    char hash[41];
    std::copy(digest.begin(), digest.end(), hash);
    hash[40] = '\0';
    return exec_git({"git", "cat-file", "-p", hash, nullptr});
}

HexDigest write_object(const std::string& type, const std::vector<uint8_t>& contents) {
    int to_child[2], from_child[2];
    if (pipe(to_child) < 0 || pipe(from_child) < 0)
        throw std::runtime_error("pipe failed");

    pid_t pid = fork();
    if (pid < 0) throw std::runtime_error("fork failed");

    if (pid == 0) {
        close(to_child[1]); close(from_child[0]);
        dup2(to_child[0], STDIN_FILENO);
        dup2(from_child[1], STDOUT_FILENO);
        close(to_child[0]); close(from_child[1]);
        int devnull = open("/dev/null", O_WRONLY);
        if (devnull >= 0) { dup2(devnull, STDERR_FILENO); close(devnull); }
        execlp("git", "git", "hash-object", "-w", "-t", type.c_str(), "--stdin", nullptr);
        _exit(1);
    }

    close(to_child[0]); close(from_child[1]);

    auto* ptr = contents.data();
    size_t rem = contents.size();
    while (rem > 0) {
        auto n = ::write(to_child[1], ptr, rem);
        if (n <= 0) break;
        ptr += n; rem -= n;
    }
    close(to_child[1]);

    char buf[64];
    ssize_t total = 0;
    while (total < 40) {
        auto n = ::read(from_child[0], buf + total, 40 - total);
        if (n <= 0) break;
        total += n;
    }
    close(from_child[0]);
    waitpid(pid, nullptr, 0);

    if (total < 40) throw std::runtime_error("bad hash-object output");
    HexDigest d;
    std::copy(buf, buf + 40, d.begin());
    return d;
}

void update_reference(const std::string& ref, const std::string& hash) {
    exec_git({"git", "update-ref", ref.c_str(), hash.c_str(), nullptr});
}

}  // namespace git
