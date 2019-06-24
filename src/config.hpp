#pragma once

#ifndef PROJECT_NAME
#define PROJECT_NAME "tensorcore_bench"
#endif /* PROJECT_NAME */

#define CUDA_MAX_GRID_SIZE static_cast<size_t>(2147483647ull)
#define CUDA_MAX_GRID_SIZE_DIV_2 (CUDA_MAX_GRID_SIZE / 2)
