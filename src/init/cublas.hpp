#pragma once

#include <cublas_v2.h>

extern cublasHandle_t cublas_handle;

bool init_cublas();
