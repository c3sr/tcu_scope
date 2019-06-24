#pragma once

#include <iostream>

#include "config.hpp"

#include "utils/commandlineflags.hpp"

DECLARE_int32(cuda_device_id);
DECLARE_bool(help);

extern void init_flags(int argc, char **argv);
