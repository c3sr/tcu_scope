#pragma once

#include "spdlog/spdlog.h"

namespace bench {
namespace init {
  namespace logger {
    extern std::shared_ptr<spdlog::logger> console;
  }
} // namespace init
} // namespace bench

#define LOG(level, ...) bench::init::logger::console->level(__VA_ARGS__)
