#pragma once

#include "types.h"
#include <optional>
#include <string>

std::optional<SolveResult> solve(const ObjectTemplate& tpl,
                                const std::string& prefix_hex,
                                const std::string& backend = "");
