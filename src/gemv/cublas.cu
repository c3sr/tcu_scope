
#include <benchmark/benchmark.h>

#include "gemv/args.hpp"
#include "init/init.hpp"
#include "utils/utils.hpp"

/* y = alpha * Ax + beta * y
cublasStatus_t cublasSgemv(cublasHandle_t handle, cublasOperation_t trans,
                           int m, int n,
                           const float           *alpha,
                           const float           *A, int lda,
                           const float           *x, int incx,
                           const float           *beta,
                           float           *y, int incy) */
static void CUBLAS_GEMV(benchmark::State &state) {

  const auto M_GLOBAL = state.range(0);
  const auto N_GLOBAL = state.range(1);

  const float alpha = 1.1f;
  const float beta  = 1.2f;

  float *a_fp32;
  float *x_fp32;
  float *y_fp32;

  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp32, M_GLOBAL * N_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &x_fp32, N_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &y_fp32, M_GLOBAL * sizeof(float)));

  curandGenerator_t gen;

  PRINT_IF_ERROR(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  PRINT_IF_ERROR(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, a_fp32, M_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, x_fp32, N_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, y_fp32, M_GLOBAL));

  PRINT_IF_ERROR(curandDestroyGenerator(gen));

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  cublasHandle_t cublasHandle;
  PRINT_IF_ERROR(cublasCreate(&cublasHandle));
  // Not use tensor cores
  PRINT_IF_ERROR(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH));

  const int incx = 1;
  const int incy = 1;

  for (auto _ : state) {
    PRINT_IF_ERROR(cudaEventRecord(start));

    PRINT_IF_ERROR(cublasSgemv(cublasHandle, CUBLAS_OP_N, M_GLOBAL, N_GLOBAL, &alpha,
                               a_fp32, M_GLOBAL, x_fp32, incx, &beta, y_fp32, incy));

    PRINT_IF_ERROR(cudaEventRecord(stop));
    PRINT_IF_ERROR(cudaEventSynchronize(stop));

    state.PauseTiming();

    float msecTotal = 0.0f;
    PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
    state.SetIterationTime(msecTotal / 1000);
    state.ResumeTiming();
  }

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  PRINT_IF_ERROR(cudaFree(a_fp32));
  PRINT_IF_ERROR(cudaFree(x_fp32));
  PRINT_IF_ERROR(cudaFree(y_fp32));

  cudaDeviceReset();

  state.counters.insert({{"M", M_GLOBAL},
                         {"N", N_GLOBAL},
                         {"num_elements", M_GLOBAL * N_GLOBAL},
                         {"flops",
                          {state.iterations() * 2.0 * M_GLOBAL * N_GLOBAL,
                           benchmark::Counter::kAvgThreadsRate}}});
}

BENCHMARK(CUBLAS_GEMV)->ARGS()->UseManualTime();
