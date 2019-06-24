#pragma once

#include <cuda_runtime.h>

#include "utils/utils.hpp"

extern bool has_cuda;

extern float device_giga_bandwidth;
extern size_t device_free_physmem;
extern size_t device_total_physmem;
extern int num_cuda_sms;

extern cudaDeviceProp cuda_device_prop;

bool init_cuda();
