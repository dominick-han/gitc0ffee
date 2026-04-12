#pragma once

#include "types.h"
#include <optional>

std::optional<SolveResult> gpu_solve(const ObjectTemplate& tpl,
                                     const std::string& prefix_hex);
