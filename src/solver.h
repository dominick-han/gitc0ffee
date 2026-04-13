#pragma once

#include "types.h"
#include <optional>

std::optional<SolveResult> solve(const ObjectTemplate& tpl,
                                const std::string& prefix_hex);
