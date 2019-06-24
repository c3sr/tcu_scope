#include "config.hpp"

#include "logger.hpp"

#define XSTRINGIFY(s) STRINGIFY(s)
#define STRINGIFY(s) #s

std::shared_ptr<spdlog::logger> bench::init::logger::console =
    spdlog::stdout_logger_mt(XSTRINGIFY(PROJECT_NAME));
