
#include "utils/utils.hpp"

DEFINE_int32(cuda_device_id, 1, "The cuda device id to use.");
// DEFINE_bool(fast, false, "Whether to run only parts of the tests.");
DEFINE_int32(verbose, 1, "Verbose level.");
DEFINE_bool(help, false, "Show help message.");

static void parse(int* argc, char** argv) {
  using namespace utils;
  for (int i = 1; i < *argc; ++i) {
    if (ParseInt32Flag(argv[i], "cuda_device_id", &FLAG(cuda_device_id)) ||
        ParseBoolFlag(argv[i], "h", &FLAG(help)) ||
        ParseBoolFlag(argv[i], "help", &FLAG(help)) ||
        ParseInt32Flag(argv[i], "v", &FLAG(verbose))) {
      for (int j = i; j != *argc - 1; ++j)
        argv[j] = argv[j + 1];

      --(*argc);
      --i;
    }
  }
}

void init_flags(int argc, char** argv) {
  parse(&argc, argv);

  return;
}
