
#include <benchmark/benchmark.h>

#include "gemm/args.hpp"
#include "init/init.hpp"
#include "utils/utils.hpp"

#include <mma.h>
using namespace nvcuda;

static void CUBLAS_WMMA_GEMM(benchmark::State &state) {
  if (!has_cuda) {
    state.SkipWithError(fmt::format("CUBLAS_WMMA_GEMM no CUDA device found").c_str());
    return;
  }

  const auto M_GLOBAL = state.range(0);
  const auto N_GLOBAL = state.range(1);
  const auto K_GLOBAL = state.range(2);

  const float alpha = 1.1f;
  const float beta  = 1.2f;

  float *a_fp32;
  float *b_fp32;
  half *a_fp16;
  half *b_fp16;
  float *c_fp32;

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  curandGenerator_t gen;
  cublasHandle_t cublasHandle;
  PRINT_IF_ERROR(cublasCreate(&cublasHandle));
  PRINT_IF_ERROR(cublasSetMathMode(cublasHandle,
                                   CUBLAS_TENSOR_OP_MATH)); // Use tensor cores

  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp32, M_GLOBAL * K_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp32, K_GLOBAL * N_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp16, M_GLOBAL * K_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp16, K_GLOBAL * N_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &c_fp32, M_GLOBAL * N_GLOBAL * sizeof(float)));

  PRINT_IF_ERROR(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  PRINT_IF_ERROR(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

  PRINT_IF_ERROR(curandGenerateUniform(gen, a_fp32, M_GLOBAL * K_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, b_fp32, K_GLOBAL * N_GLOBAL));

  // curand doesn't currently support fp16 so we generate in fp32 and convert to
  // fp16.
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(M_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(
      a_fp16, a_fp32, M_GLOBAL * K_GLOBAL)));
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(K_GLOBAL * N_GLOBAL + 255) / 256, 256>>>(
      b_fp16, b_fp32, K_GLOBAL * N_GLOBAL)));

  PRINT_IF_ERROR(curandGenerateUniform(gen, c_fp32, M_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(curandDestroyGenerator(gen));

  for (auto _ : state) {
    PRINT_IF_ERROR(cudaEventRecord(start));

    PRINT_IF_ERROR(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, M_GLOBAL,
                                N_GLOBAL, K_GLOBAL, &alpha, a_fp16, CUDA_R_16F, M_GLOBAL,
                                b_fp16, CUDA_R_16F, K_GLOBAL, &beta, c_fp32, CUDA_R_32F,
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
  PRINT_IF_ERROR(cudaFree(b_fp32));
  PRINT_IF_ERROR(cudaFree(a_fp16));
  PRINT_IF_ERROR(cudaFree(b_fp16));
  PRINT_IF_ERROR(cudaFree(c_fp32));

  cudaDeviceReset();

  state.counters.insert({{"M", M_GLOBAL},
                         {"N", N_GLOBAL},
                         {"K", K_GLOBAL},
                         {"num_elements", M_GLOBAL * N_GLOBAL * K_GLOBAL},
                         {"flops",
                          {state.iterations() * 2.0 * M_GLOBAL * N_GLOBAL * K_GLOBAL,
                           benchmark::Counter::kAvgThreadsRate}}});
}

static void CUBLAS_GEMM(benchmark::State &state) {
  if (!has_cuda) {
    state.SkipWithError(fmt::format("CUBLAS_GEMM no CUDA device found").c_str());
    return;
  }

  const auto M_GLOBAL = state.range(0);
  const auto N_GLOBAL = state.range(1);
  const auto K_GLOBAL = state.range(2);

  const float alpha = 1.1f;
  const float beta  = 1.2f;

  float *a_fp32;
  float *b_fp32;
  half *a_fp16;
  half *b_fp16;
  float *c_fp32;

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  curandGenerator_t gen;
  cublasHandle_t cublasHandle;

  PRINT_IF_ERROR(cublasCreate(&cublasHandle));

  // Not use tensor cores
  PRINT_IF_ERROR(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH));

  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp32, M_GLOBAL * K_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp32, K_GLOBAL * N_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp16, M_GLOBAL * K_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp16, K_GLOBAL * N_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &c_fp32, M_GLOBAL * N_GLOBAL * sizeof(float)));

  PRINT_IF_ERROR(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  PRINT_IF_ERROR(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

  PRINT_IF_ERROR(curandGenerateUniform(gen, a_fp32, M_GLOBAL * K_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, b_fp32, K_GLOBAL * N_GLOBAL));

  // curand doesn't currently support fp16 so we generate in fp32 and convert to
  // fp16.
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(M_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(
      a_fp16, a_fp32, M_GLOBAL * K_GLOBAL)));
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(K_GLOBAL * N_GLOBAL + 255) / 256, 256>>>(
      b_fp16, b_fp32, K_GLOBAL * N_GLOBAL)));

  PRINT_IF_ERROR(curandGenerateUniform(gen, c_fp32, M_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(curandDestroyGenerator(gen));

  // printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", M_GLOBAL,
  // N_GLOBAL, K_GLOBAL, alpha, beta);

  for (auto _ : state) {
    PRINT_IF_ERROR(cudaEventRecord(start));

    PRINT_IF_ERROR(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, M_GLOBAL,
                                N_GLOBAL, K_GLOBAL, &alpha, a_fp16, CUDA_R_16F, M_GLOBAL,
                                b_fp16, CUDA_R_16F, K_GLOBAL, &beta, c_fp32, CUDA_R_32F,
                                M_GLOBAL, CUDA_R_32F, CUBLAS_GEMM_DEFAULT));

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
  PRINT_IF_ERROR(cudaFree(b_fp32));
  PRINT_IF_ERROR(cudaFree(a_fp16));
  PRINT_IF_ERROR(cudaFree(b_fp16));
  PRINT_IF_ERROR(cudaFree(c_fp32));

  cudaDeviceReset();

  state.counters.insert({{"M", M_GLOBAL},
                         {"N", N_GLOBAL},
                         {"K", K_GLOBAL},
                         {"num_elements", M_GLOBAL * N_GLOBAL * K_GLOBAL},
                         {"flops",
                          {state.iterations() * 2.0 * M_GLOBAL * N_GLOBAL * K_GLOBAL,
                           benchmark::Counter::kAvgThreadsRate}}});
}

static void CUBLAS_WMMA_HGEMM(benchmark::State &state) {

  const auto M_GLOBAL = state.range(0);
  const auto N_GLOBAL = state.range(1);
  const auto K_GLOBAL = state.range(2);

  const __half alpha = approx_float_to_half(1.1f);
  const __half beta  = approx_float_to_half(1.2f);

  float *a_fp32;
  float *b_fp32;
  half *a_fp16;
  half *b_fp16;

  float *c_fp32;
  half *c_fp16;

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  curandGenerator_t gen;
  cublasHandle_t cublasHandle;

  PRINT_IF_ERROR(cublasCreate(&cublasHandle));

  // Use tensor cores
  PRINT_IF_ERROR(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp32, M_GLOBAL * K_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp32, K_GLOBAL * N_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &c_fp32, M_GLOBAL * N_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp16, M_GLOBAL * K_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp16, K_GLOBAL * N_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &c_fp16, M_GLOBAL * N_GLOBAL * sizeof(half)));

  PRINT_IF_ERROR(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  PRINT_IF_ERROR(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

  PRINT_IF_ERROR(curandGenerateUniform(gen, a_fp32, M_GLOBAL * K_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, b_fp32, K_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, c_fp32, M_GLOBAL * N_GLOBAL));

  // curand doesn't currently support fp16 so we generate in fp32 and convert to
  // fp16.
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(M_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(
      a_fp16, a_fp32, M_GLOBAL * K_GLOBAL)));
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(K_GLOBAL * N_GLOBAL + 255) / 256, 256>>>(
      b_fp16, b_fp32, K_GLOBAL * N_GLOBAL)));
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(M_GLOBAL * N_GLOBAL + 255) / 256, 256>>>(
      c_fp16, c_fp32, M_GLOBAL * N_GLOBAL)));

  PRINT_IF_ERROR(curandGenerateUniform(gen, c_fp32, M_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(curandDestroyGenerator(gen));

  // printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", M_GLOBAL,
  // N_GLOBAL, K_GLOBAL, alpha, beta);

  const auto lda = M_GLOBAL;
  const auto ldb = K_GLOBAL;
  const auto ldc = M_GLOBAL;

  for (auto _ : state) {
    PRINT_IF_ERROR(cudaEventRecord(start));

    PRINT_IF_ERROR(cublasHgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, M_GLOBAL, N_GLOBAL,
                               K_GLOBAL, &alpha, a_fp16, lda, b_fp16, ldb, &beta, c_fp16,
                               ldc));
    // PRINT_IF_ERROR(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
    // M_GLOBAL, N_GLOBAL, K_GLOBAL, &alpha, a_fp16,
    //                             CUDA_R_16F, M_GLOBAL, b_fp16, CUDA_R_16F,
    //                             K_GLOBAL, &beta, c_fp16, CUDA_R_16F,
    //                             M_GLOBAL, CUDA_R_16F,
    //                             CUBLAS_GEMM_DFALT_TENSOR_OP));

    PRINT_IF_ERROR(cudaEventRecord(stop));
    PRINT_IF_ERROR(cudaEventSynchronize(stop));

    state.PauseTiming();

    float msecTotal = 0.0f;
    PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
    state.SetIterationTime(msecTotal / 1000);
    state.ResumeTiming();
  }

  state.counters.insert({{"M", M_GLOBAL},
                         {"N", N_GLOBAL},
                         {"K", K_GLOBAL},
                         {"num_elements", M_GLOBAL * N_GLOBAL * K_GLOBAL},                   
                         {"flops",
                          {state.iterations() * 2.0 * M_GLOBAL * N_GLOBAL * K_GLOBAL,
                           benchmark::Counter::kAvgThreadsRate}}});

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  PRINT_IF_ERROR(cudaFree(a_fp32));
  PRINT_IF_ERROR(cudaFree(b_fp32));
  PRINT_IF_ERROR(cudaFree(c_fp32));
  PRINT_IF_ERROR(cudaFree(a_fp16));
  PRINT_IF_ERROR(cudaFree(b_fp16));
  PRINT_IF_ERROR(cudaFree(c_fp16));

  cudaDeviceReset();
}

static void CUBLAS_HGEMM(benchmark::State &state) {
  if (!has_cuda) {
    state.SkipWithError(fmt::format("CUBLAS_HGEMM no CUDA device found").c_str());
    return;
  }

  const auto M_GLOBAL = state.range(0);
  const auto N_GLOBAL = state.range(1);
  const auto K_GLOBAL = state.range(2);

  const __half alpha = approx_float_to_half(1.1f);
  const __half beta  = approx_float_to_half(1.2f);

  float *a_fp32;
  float *b_fp32;
  half *a_fp16;
  half *b_fp16;

  float *c_fp32;
  half *c_fp16;

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  curandGenerator_t gen;
  cublasHandle_t cublasHandle;

  PRINT_IF_ERROR(cublasCreate(&cublasHandle));

  // Not use tensor cores
  PRINT_IF_ERROR(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH));

  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp32, M_GLOBAL * K_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp32, K_GLOBAL * N_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &c_fp32, M_GLOBAL * N_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp16, M_GLOBAL * K_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp16, K_GLOBAL * N_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &c_fp16, M_GLOBAL * N_GLOBAL * sizeof(half)));

  PRINT_IF_ERROR(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  PRINT_IF_ERROR(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

  PRINT_IF_ERROR(curandGenerateUniform(gen, a_fp32, M_GLOBAL * K_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, b_fp32, K_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, c_fp32, M_GLOBAL * N_GLOBAL));

  // curand doesn't currently support fp16 so we generate in fp32 and convert to
  // fp16.
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(M_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(
      a_fp16, a_fp32, M_GLOBAL * K_GLOBAL)));
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(K_GLOBAL * N_GLOBAL + 255) / 256, 256>>>(
      b_fp16, b_fp32, K_GLOBAL * N_GLOBAL)));
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(M_GLOBAL * N_GLOBAL + 255) / 256, 256>>>(
      c_fp16, c_fp32, M_GLOBAL * N_GLOBAL)));

  PRINT_IF_ERROR(curandGenerateUniform(gen, c_fp32, M_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(curandDestroyGenerator(gen));

  // printf("\nM = %d, N = %d, K = %d. alpha = %f, beta = %f\n\n", M_GLOBAL,
  // N_GLOBAL, K_GLOBAL, alpha, beta);

  const auto lda = M_GLOBAL;
  const auto ldb = K_GLOBAL;
  const auto ldc = M_GLOBAL;

  for (auto _ : state) {
    PRINT_IF_ERROR(cudaEventRecord(start));

    PRINT_IF_ERROR(cublasHgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, M_GLOBAL, N_GLOBAL,
                               K_GLOBAL, &alpha, a_fp16, lda, b_fp16, ldb, &beta, c_fp16,
                               ldc));
    // PRINT_IF_ERROR(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
    // M_GLOBAL, N_GLOBAL, K_GLOBAL, &alpha, a_fp16,
    //                             CUDA_R_16F, M_GLOBAL, b_fp16, CUDA_R_16F,
    //                             K_GLOBAL, &beta, c_fp16, CUDA_R_16F,
    //                             M_GLOBAL, CUDA_R_16F,
    //                             CUBLAS_GEMM_DFALT_TENSOR_OP));

    PRINT_IF_ERROR(cudaEventRecord(stop));
    PRINT_IF_ERROR(cudaEventSynchronize(stop));

    state.PauseTiming();

    float msecTotal = 0.0f;
    PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
    state.SetIterationTime(msecTotal / 1000);
    state.ResumeTiming();
  }

  state.counters.insert({{"M", M_GLOBAL},
                         {"N", N_GLOBAL},
                         {"K", K_GLOBAL},
                         {"num_elements", M_GLOBAL * N_GLOBAL * K_GLOBAL},                      
                         {"flops",
                          {state.iterations() * 2.0 * M_GLOBAL * N_GLOBAL * K_GLOBAL,
                           benchmark::Counter::kAvgThreadsRate}}});

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  PRINT_IF_ERROR(cudaFree(a_fp32));
  PRINT_IF_ERROR(cudaFree(b_fp32));
  PRINT_IF_ERROR(cudaFree(c_fp32));
  PRINT_IF_ERROR(cudaFree(a_fp16));
  PRINT_IF_ERROR(cudaFree(b_fp16));
  PRINT_IF_ERROR(cudaFree(c_fp16));

  cudaDeviceReset();
}

static void CUBLAS_WMMA_HGEMM_GemmEx(benchmark::State &state) {


  const auto M_GLOBAL = state.range(0);
  const auto N_GLOBAL = state.range(1);
  const auto K_GLOBAL = state.range(2);

  const __half alpha = approx_float_to_half(1.1f);
  const __half beta  = approx_float_to_half(1.2f);

  float *a_fp32;
  float *b_fp32;
  half *a_fp16;
  half *b_fp16;

  float *c_fp32;
  half *c_fp16;

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  curandGenerator_t gen;
  cublasHandle_t cublasHandle;

  PRINT_IF_ERROR(cublasCreate(&cublasHandle));

  // Use tensor cores
  PRINT_IF_ERROR(cublasSetMathMode(cublasHandle, CUBLAS_TENSOR_OP_MATH));

  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp32, M_GLOBAL * K_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp32, K_GLOBAL * N_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &c_fp32, M_GLOBAL * N_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp16, M_GLOBAL * K_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp16, K_GLOBAL * N_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &c_fp16, M_GLOBAL * N_GLOBAL * sizeof(half)));

  PRINT_IF_ERROR(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  PRINT_IF_ERROR(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

  PRINT_IF_ERROR(curandGenerateUniform(gen, a_fp32, M_GLOBAL * K_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, b_fp32, K_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, c_fp32, M_GLOBAL * N_GLOBAL));

  // curand doesn't currently support fp16 so we generate in fp32 and convert to
  // fp16.
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(M_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(
      a_fp16, a_fp32, M_GLOBAL * K_GLOBAL)));
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(K_GLOBAL * N_GLOBAL + 255) / 256, 256>>>(
      b_fp16, b_fp32, K_GLOBAL * N_GLOBAL)));
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(M_GLOBAL * N_GLOBAL + 255) / 256, 256>>>(
      c_fp16, c_fp32, M_GLOBAL * N_GLOBAL)));

  PRINT_IF_ERROR(curandGenerateUniform(gen, c_fp32, M_GLOBAL * N_GLOBAL));

  PRINT_IF_ERROR(curandDestroyGenerator(gen));

  for (auto _ : state) {
    PRINT_IF_ERROR(cudaEventRecord(start));

    // PRINT_IF_ERROR(cublasHgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
    // M_GLOBAL, N_GLOBAL, K_GLOBAL, &alpha, a_fp16, lda, b_fp16, ldb, &beta,
    // c_fp16, ldc));
    PRINT_IF_ERROR(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, M_GLOBAL,
                                N_GLOBAL, K_GLOBAL, &alpha, a_fp16, CUDA_R_16F, M_GLOBAL,
                                b_fp16, CUDA_R_16F, K_GLOBAL, &beta, c_fp16, CUDA_R_16F,
                                M_GLOBAL, CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP));

    PRINT_IF_ERROR(cudaEventRecord(stop));
    PRINT_IF_ERROR(cudaEventSynchronize(stop));

    state.PauseTiming();

    float msecTotal = 0.0f;
    PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
    state.SetIterationTime(msecTotal / 1000);
    state.ResumeTiming();
  }

  state.counters.insert({{"M", M_GLOBAL},
                         {"N", N_GLOBAL},
                         {"K", K_GLOBAL},
                         {"num_elements", M_GLOBAL * N_GLOBAL * K_GLOBAL},                        
                         {"flops",
                          {state.iterations() * 2.0 * M_GLOBAL * N_GLOBAL * K_GLOBAL,
                           benchmark::Counter::kAvgThreadsRate}}});

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  PRINT_IF_ERROR(cudaFree(a_fp32));
  PRINT_IF_ERROR(cudaFree(b_fp32));
  PRINT_IF_ERROR(cudaFree(c_fp32));
  PRINT_IF_ERROR(cudaFree(a_fp16));
  PRINT_IF_ERROR(cudaFree(b_fp16));
  PRINT_IF_ERROR(cudaFree(c_fp16));

  cudaDeviceReset();
}

static void CUBLAS_HGEMM_GemmEx(benchmark::State &state) {
  if (!has_cuda) {
    state.SkipWithError(fmt::format("CUBLAS_HGEMM_GemmEx no CUDA device found").c_str());
    return;
  }

  const auto M_GLOBAL = state.range(0);
  const auto N_GLOBAL = state.range(1);
  const auto K_GLOBAL = state.range(2);

  const __half alpha = approx_float_to_half(1.1f);
  const __half beta  = approx_float_to_half(1.2f);

  float *a_fp32;
  float *b_fp32;
  half *a_fp16;
  half *b_fp16;

  float *c_fp32;
  half *c_fp16;

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  curandGenerator_t gen;
  cublasHandle_t cublasHandle;

  PRINT_IF_ERROR(cublasCreate(&cublasHandle));

  // Not use tensor cores
  PRINT_IF_ERROR(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH));

  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp32, M_GLOBAL * K_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp32, K_GLOBAL * N_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &c_fp32, M_GLOBAL * N_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp16, M_GLOBAL * K_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp16, K_GLOBAL * N_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &c_fp16, M_GLOBAL * N_GLOBAL * sizeof(half)));

  PRINT_IF_ERROR(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  PRINT_IF_ERROR(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

  PRINT_IF_ERROR(curandGenerateUniform(gen, a_fp32, M_GLOBAL * K_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, b_fp32, K_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, c_fp32, M_GLOBAL * N_GLOBAL));

  // curand doesn't currently support fp16 so we generate in fp32 and convert to
  // fp16.
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(M_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(
      a_fp16, a_fp32, M_GLOBAL * K_GLOBAL)));
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(K_GLOBAL * N_GLOBAL + 255) / 256, 256>>>(
      b_fp16, b_fp32, K_GLOBAL * N_GLOBAL)));
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(M_GLOBAL * N_GLOBAL + 255) / 256, 256>>>(
      c_fp16, c_fp32, M_GLOBAL * N_GLOBAL)));

  PRINT_IF_ERROR(curandGenerateUniform(gen, c_fp32, M_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(curandDestroyGenerator(gen));

  for (auto _ : state) {
    PRINT_IF_ERROR(cudaEventRecord(start));

    // PRINT_IF_ERROR(cublasHgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
    // M_GLOBAL, N_GLOBAL, K_GLOBAL, &alpha, a_fp16, lda, b_fp16, ldb, &beta,
    // c_fp16, ldc));
    PRINT_IF_ERROR(cublasGemmEx(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, M_GLOBAL,
                                N_GLOBAL, K_GLOBAL, &alpha, a_fp16, CUDA_R_16F, M_GLOBAL,
                                b_fp16, CUDA_R_16F, K_GLOBAL, &beta, c_fp16, CUDA_R_16F,
                                M_GLOBAL, CUDA_R_16F, CUBLAS_GEMM_DEFAULT));

    PRINT_IF_ERROR(cudaEventRecord(stop));
    PRINT_IF_ERROR(cudaEventSynchronize(stop));

    state.PauseTiming();

    float msecTotal = 0.0f;
    PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));

    state.SetIterationTime(msecTotal / 1000);
    state.ResumeTiming();
  }

  state.counters.insert({{"M", M_GLOBAL},
                         {"N", N_GLOBAL},
                         {"K", K_GLOBAL},
                         {"num_elements", M_GLOBAL * N_GLOBAL * K_GLOBAL},                      
                         {"flops",
                          {state.iterations() * 2.0 * M_GLOBAL * N_GLOBAL * K_GLOBAL,
                           benchmark::Counter::kAvgThreadsRate}}});

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  PRINT_IF_ERROR(cudaFree(a_fp32));
  PRINT_IF_ERROR(cudaFree(b_fp32));
  PRINT_IF_ERROR(cudaFree(c_fp32));
  PRINT_IF_ERROR(cudaFree(a_fp16));
  PRINT_IF_ERROR(cudaFree(b_fp16));
  PRINT_IF_ERROR(cudaFree(c_fp16));

  cudaDeviceReset();
}

// 16F input and 32F compute
BENCHMARK(CUBLAS_GEMM)->ARGS()->UseManualTime();
BENCHMARK(CUBLAS_WMMA_GEMM)->ARGS()->UseManualTime();

// 16F input and 16F compute using cublashgemm()
BENCHMARK(CUBLAS_HGEMM)->ARGS()->UseManualTime();
BENCHMARK(CUBLAS_WMMA_HGEMM)->ARGS()->UseManualTime();

// 16F input and 16F compute using cublasgemmex()
BENCHMARK(CUBLAS_WMMA_HGEMM_GemmEx)->ARGS()->UseManualTime();
BENCHMARK(CUBLAS_HGEMM_GemmEx)->ARGS()->UseManualTime();
