#pragma once

#include "../types.h"

struct CommitObject {
    std::vector<std::string> headers;
    std::string message;
};

CommitObject parse_commit(const std::vector<uint8_t>& payload);
ObjectTemplate prepare_template(const CommitObject& obj);
