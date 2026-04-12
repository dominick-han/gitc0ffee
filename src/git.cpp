#include "git.h"

#include <array>
#include <cstdio>
#include <fcntl.h>
#include <memory>
#include <stdexcept>
#include <sys/wait.h>
#include <unistd.h>

namespace git {

// Run a git command, return stdout with trailing newlines stripped.
static std::string run(const std::string& args) {
    std::string cmd = args + " 2>/dev/null";
    std::unique_ptr<FILE, decltype(&pclose)> p(popen(cmd.c_str(), "r"), pclose);
    if (!p) throw std::runtime_error("popen failed");
    std::string out;
    std::array<char, 4096> buf;
    while (auto n = fread(buf.data(), 1, buf.size(), p.get()))
        out.append(buf.data(), n);
    while (!out.empty() && out.back() == '\n') out.pop_back();
    return out;
}

HexDigest get_head_digest() {
    auto out = run("git rev-parse HEAD");
    if (out.size() != 40) throw std::runtime_error("bad HEAD: " + out);
    HexDigest d;
    std::copy(out.begin(), out.end(), d.begin());
    return d;
}

std::vector<uint8_t> get_commit_contents(const HexDigest& digest) {
    std::string cmd = "git cat-file -p ";
    cmd.append(digest.data(), 40);
    cmd += " 2>/dev/null";
    std::unique_ptr<FILE, decltype(&pclose)> p(popen(cmd.c_str(), "r"), pclose);
    if (!p) throw std::runtime_error("popen failed");
    std::vector<uint8_t> out;
    std::array<char, 4096> buf;
    while (auto n = fread(buf.data(), 1, buf.size(), p.get()))
        out.insert(out.end(), buf.data(), buf.data() + n);
    return out;
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
    run("git update-ref '" + ref + "' '" + hash + "'");
}

}  // namespace git
