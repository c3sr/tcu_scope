
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "utils/utils.hpp"

cublasHandle_t cublas_handle;

bool init_cublas() {
  return PRINT_IF_ERROR(cublasCreate(&cublas_handle));
}
