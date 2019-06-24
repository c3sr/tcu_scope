
#pragma once

#include <cublas_v2.h>
#include <curand.h>
#include <stdio.h>

// Define some error checking macros.
#define checkKernelErrors(expr)                                                          \
  do {                                                                                   \
    expr;                                                                                \
                                                                                         \
    cudaError_t __err = cudaGetLastError();                                              \
    if (__err != cudaSuccess) {                                                          \
      printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err));  \
      abort();                                                                           \
    }                                                                                    \
  } while (0)

#define cudaErrCheck(stat)                                                               \
  { cudaErrCheck_((stat), __FILE__, __LINE__); }
static void cudaErrCheck_(cudaError_t stat, const char *file, int line) {
  if (stat != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(stat), file, line);
  }
}

#define cublasErrCheck(stat)                                                             \
  { cublasErrCheck_((stat), __FILE__, __LINE__); }
static void cublasErrCheck_(cublasStatus_t stat, const char *file, int line) {
  if (stat != CUBLAS_STATUS_SUCCESS) {
    fprintf(stderr, "cuBLAS Error: %d %s %d\n", stat, file, line);
  }
}

#define curandErrCheck(stat)                                                             \
  { curandErrCheck_((stat), __FILE__, __LINE__); }
static void curandErrCheck_(curandStatus_t stat, const char *file, int line) {
  if (stat != CURAND_STATUS_SUCCESS) {
    fprintf(stderr, "cuRand Error: %d %s %d\n", stat, file, line);
  }
}
