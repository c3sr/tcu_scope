#pragma once

#include <chrono>

#include "utils/compat.hpp"

static ALWAYS_INLINE std::chrono::time_point<std::chrono::high_resolution_clock> now() {
  return std::chrono::high_resolution_clock::now();
}