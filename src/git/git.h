#pragma once

#include "../types.h"
#include <vector>

namespace git {

HexDigest get_head_digest();
std::vector<uint8_t> get_commit_contents(const HexDigest& digest);
HexDigest write_object(const std::string& type, const std::vector<uint8_t>& contents);
void update_reference(const std::string& ref, const std::string& hash);

}  // namespace git
