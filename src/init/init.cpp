#include "init/init.hpp"

void init(int argc, char **argv) {
  bench::init::logger::console = spdlog::stdout_logger_mt(argv[0]);
  init_flags(argc, argv);
  init_cuda();
  init_cublas();
}
