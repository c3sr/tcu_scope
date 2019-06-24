#pragma once

#include "config.hpp"

#include <cuda_runtime.h>

#include "init/cublas.hpp"
#include "init/cuda.hpp"
#include "init/flags.hpp"
#include "init/logger.hpp"

void init(int argc, char **argv);
