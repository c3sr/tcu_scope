
#include <benchmark/benchmark.h>

#include "gemv/args.hpp"
#include "init/init.hpp"
#include "utils/utils.hpp"

#include <mma.h>
using namespace nvcuda;

// MMA matrix tile dimensions. (16, 16, 16), (32, 8, 16), and (8, 32, 16) are
// currently supported.
// static const int M = 16;
static const int N = 16;
// static const int K = 16;

static void CUDA_WMMA_GEMV_CUBLAS(benchmark::State &state) {
  const auto M_GLOBAL = state.range(0);
  const auto K_GLOBAL = state.range(1);
  const auto N_GLOBAL = N;

  const float alpha = 1.1f;
  const float beta  = 1.2f;

  float *a_fp32;
  float *x_fp32;
  half *a_fp16;
  half *x_fp16;
  half *b_fp16;

  float *y;

  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp32, M_GLOBAL * K_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &x_fp32, K_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp16, M_GLOBAL * K_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &x_fp16, K_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp16, K_GLOBAL * N_GLOBAL * sizeof(half)));

  PRINT_IF_ERROR(cudaMalloc((void **) &y,
                            M_GLOBAL * N_GLOBAL *
                                sizeof(float))); // the first column holds the result

  curandGenerator_t gen;
  PRINT_IF_ERROR(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  PRINT_IF_ERROR(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, a_fp32, M_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, x_fp32, N_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, y, M_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(curandDestroyGenerator(gen));

  // curand doesn't currently support fp16 so we generate in fp32 and convert to
  // fp16.
  convertFp32ToFp16<<<(M_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(a_fp16, a_fp32,
                                                                M_GLOBAL * K_GLOBAL);
  convertFp32ToFp16<<<(N_GLOBAL + 255) / 256, 256>>>(x_fp16, x_fp32, K_GLOBAL);

  // copy vector x to matrix b, column-major
  PRINT_IF_ERROR(
      cudaMemcpy(b_fp16, x_fp16, K_GLOBAL * sizeof(half), cudaMemcpyDeviceToDevice));

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  cublasHandle_t cublasHandle;
  PRINT_IF_ERROR(cublasCreate(&cublasHandle));
  PRINT_IF_ERROR(cublasSetMathMode(cublasHandle,
                                   CUBLAS_TENSOR_OP_MATH)); // Use tensor cores

  for (auto _ : state) {
    PRINT_IF_ERROR(cudaEventRecord(start));

    /* C = α op ( A ) op ( B ) + β C
      cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                               cublasOperation_t transa,
                               cublasOperation_t transb,
                               int m,
                               int n,
                               int k,
                               const void    *alpha,
                               const void     *A,
                               cudaDataType_t Atype,
                               int lda,
                               const void     *B,
                               cudaDataType_t Btype,
                               int ldb,
                               const void    *beta,
                               void           *C,
                               cudaDataType_t Ctype,
                               int ldc,
                               cudaDataType_t computeType,
                               cublasGemmAlgo_t algo)
                               */
    PRINT_IF_ERROR(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, M_GLOBAL,
                                N_GLOBAL, K_GLOBAL, &alpha, a_fp16, CUDA_R_16F, M_GLOBAL,
                                b_fp16, CUDA_R_16F, K_GLOBAL, &beta, y, CUDA_R_32F,
                                M_GLOBAL, CUDA_R_32F, CUBLAS_GEMM_DFALT_TENSOR_OP));

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
  PRINT_IF_ERROR(cudaFree(y));
  PRINT_IF_ERROR(cudaFree(a_fp16));
  PRINT_IF_ERROR(cudaFree(x_fp16));
  PRINT_IF_ERROR(cudaFree(b_fp16));

  cudaDeviceReset();

  state.counters.insert({{"M", M_GLOBAL},
                         {"N", K_GLOBAL},
                         {"num_elements", M_GLOBAL * K_GLOBAL},
                         {"flops",
                          {state.iterations() * 2.0 * M_GLOBAL * K_GLOBAL,
                           benchmark::Counter::kAvgThreadsRate}}});
}

static void CUDA_WMMA_HGEMV_CUBLAS(benchmark::State &state) {
  const auto M_GLOBAL = state.range(0);
  const auto K_GLOBAL = state.range(1);
  const auto N_GLOBAL = N;

  const __half alpha = approx_float_to_half(1.1f);
  const __half beta  = approx_float_to_half(1.2f);

  float *a_fp32;
  float *x_fp32;
  float *y_fp32;
  half *a_fp16;
  half *x_fp16;
  half *b_fp16;
  half *y_fp16;

  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp32, M_GLOBAL * K_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &x_fp32, K_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &y_fp32, M_GLOBAL * N_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp16, M_GLOBAL * K_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &x_fp16, K_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &y_fp16, M_GLOBAL * N_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp16, K_GLOBAL * N_GLOBAL * sizeof(half)));

  curandGenerator_t gen;
  PRINT_IF_ERROR(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  PRINT_IF_ERROR(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, a_fp32, M_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, x_fp32, N_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, y_fp32, M_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(curandDestroyGenerator(gen));

  // curand doesn't currently support fp16 so we generate in fp32 and convert to
  // fp16.
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(M_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(
      a_fp16, a_fp32, M_GLOBAL * K_GLOBAL)));
  PRINT_IF_LAUNCH_ERROR(
      (convertFp32ToFp16<<<(N_GLOBAL + 255) / 256, 256>>>(x_fp16, x_fp32, K_GLOBAL)));
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(M_GLOBAL * N_GLOBAL + 255) / 256, 256>>>(
      y_fp16, y_fp32, M_GLOBAL * N_GLOBAL)));

  // copy vector x to matrix b, column-major
  PRINT_IF_ERROR(
      cudaMemcpy(b_fp16, x_fp16, K_GLOBAL * sizeof(half), cudaMemcpyDeviceToDevice));

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  cublasHandle_t cublasHandle;
  PRINT_IF_ERROR(cublasCreate(&cublasHandle));
  PRINT_IF_ERROR(cublasSetMathMode(cublasHandle,
                                   CUBLAS_TENSOR_OP_MATH)); // Use tensor cores

  for (auto _ : state) {
    PRINT_IF_ERROR(cudaEventRecord(start));

    /* C = α op ( A ) op ( B ) + β C
      cublasStatus_t cublasGemmEx(cublasHandle_t handle,
                           cublasOperation_t transa,
                           cublasOperation_t transb,
                           int m,
                           int n,
                           int k,
                           const void    *alpha,
                           const void     *A,
                           cudaDataType_t Atype,
                           int lda,
                           const void     *B,
                           cudaDataType_t Btype,
                           int ldb,
                           const void    *beta,
                           void           *C,
                           cudaDataType_t Ctype,
                           int ldc,
                           cudaDataType_t computeType,
                           cublasGemmAlgo_t algo)
                           */
    PRINT_IF_ERROR(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, M_GLOBAL,
                                N_GLOBAL, K_GLOBAL, &alpha, a_fp16, CUDA_R_16F, M_GLOBAL,
                                b_fp16, CUDA_R_16F, K_GLOBAL, &beta, y_fp16, CUDA_R_16F,
                                M_GLOBAL, CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP));

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
  PRINT_IF_ERROR(cudaFree(a_fp16));
  PRINT_IF_ERROR(cudaFree(x_fp16));
  PRINT_IF_ERROR(cudaFree(y_fp16));
  PRINT_IF_ERROR(cudaFree(b_fp16));

  cudaDeviceReset();

  state.counters.insert({{"M", M_GLOBAL},
                         {"N", K_GLOBAL},
                         {"num_elements", M_GLOBAL * K_GLOBAL},
                         {"flops",
                          {state.iterations() * 2.0 * M_GLOBAL * K_GLOBAL,
                           benchmark::Counter::kAvgThreadsRate}}});
}

BENCHMARK(CUDA_WMMA_GEMV_CUBLAS)->ARGS()->UseManualTime();
BENCHMARK(CUDA_WMMA_HGEMV_CUBLAS)->ARGS()->UseManualTime();
