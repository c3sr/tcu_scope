// https://github.com/parallel-forall/code-samples/blob/master/posts/tensor-cores

#include <benchmark/benchmark.h>

#include "gemm/args.hpp"
#include "init/init.hpp"
#include "utils/utils.hpp"

#include <mma.h>
using namespace nvcuda;

#ifndef WARP_SIZE
#define WARP_SIZE (32)
#endif // WARP_SIZE

// MMA matrix tile dimensions. (16, 16, 16), (32, 8, 16), and (8, 32, 16) are
// currently supported.
static const int M = 16;
static const int N = 16;
static const int K = 16;

// Implementation constants.
static const int BLOCK_ROW_TILES = 4;
static const int BLOCK_COL_TILES = 4;

// Performs an MxNxK GEMM (C=alpha*A*B + beta*C) assuming:
//  1) Matrices are packed in memory.
//  2) M, N and K are multiples of 16.
//  3) Neither A nor B are transposed.
// Note: This is NOT a high performance example but is for demonstration
// purposes only
//       For a high performance code please use the GEMM provided in cuBLAS.
static __global__ void compute_gemm_naive(const half *__restrict__ a,
                                          const half *__restrict__ b, float *c,
                                          int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
                                          float alpha, float beta) {
  // Leading dimensions. Packed with no transpositions.
  int lda = M_GLOBAL;
  int ldb = K_GLOBAL;
  int ldc = M_GLOBAL;

  // Global warp id
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, M, N, K, float> acc_frag;
  wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

  wmma::fill_fragment(acc_frag, zero<float>());

  // Loop over k
  for (int i = 0; i < K_GLOBAL; i += K) {
    int aRow = warpM * M;
    int aCol = i;

    int bRow = i;
    int bCol = warpN * N;

    // Bounds checking
    if (aRow < M_GLOBAL && bCol < N_GLOBAL) {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
      wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Load in the current value of c, scale it by beta, and add this our result
  // scaled by alpha
  int cRow = warpM * M;
  int cCol = warpN * N;

  if (cRow < M && cCol < N) {
    wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);

    for (int i = 0; i < c_frag.num_elements; i++) {
      c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }

    // Store the output
    wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
  }
}

static __global__ void compute_hgemm_naive(const half *__restrict__ a,
                                           const half *__restrict__ b, half *c,
                                           int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
                                           half alpha, half beta) {
  // Leading dimensions. Packed with no transpositions.
  int lda = M_GLOBAL;
  int ldb = K_GLOBAL;
  int ldc = M_GLOBAL;

  // Global warp id
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  int warpN = (blockIdx.y * blockDim.y + threadIdx.y);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> acc_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> c_frag;

  wmma::fill_fragment(acc_frag, zero<half>());

  // Loop over k
  for (int i = 0; i < K_GLOBAL; i += K) {
    int aRow = warpM * M;
    int aCol = i;

    int bRow = i;
    int bCol = warpN * N;

    // Bounds checking
    if (aRow < M_GLOBAL && bCol < N_GLOBAL) {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
      wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Load in the current value of c, scale it by beta, and add this our result
  // scaled by alpha
  int cRow = warpM * M;
  int cCol = warpN * N;

  if (cRow < M && cCol < N) {
    wmma::load_matrix_sync(c_frag, c + cRow + cCol * ldc, ldc, wmma::mem_col_major);

    for (int i = 0; i < c_frag.num_elements; i++) {
      c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }

    // Store the output
    wmma::store_matrix_sync(c + cRow + cCol * ldc, c_frag, ldc, wmma::mem_col_major);
  }
}

static void CUDA_WMMA_GEMM_NAIVE(benchmark::State &state) {
  /* if (!has_cuda) { */
  /*   state.SkipWithError(fmt::format("CUDA_WMMA_GEMM_NAIVE no CUDA device
   * found")); */
  /*   return; */
  /* } */

  const auto M_GLOBAL = state.range(0);
  const auto N_GLOBAL = state.range(1);
  const auto K_GLOBAL = state.range(2);

  const float alpha = 1.1f;
  const float beta  = 1.2f;

  float *a_fp32;
  float *b_fp32;
  float *c;
  half *a_fp16;
  half *b_fp16;

  curandGenerator_t gen;

  // Use tensor cores
  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp32, M_GLOBAL * K_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp32, K_GLOBAL * N_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp16, M_GLOBAL * K_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp16, K_GLOBAL * N_GLOBAL * sizeof(half)));

  PRINT_IF_ERROR(cudaMalloc((void **) &c, M_GLOBAL * N_GLOBAL * sizeof(float)));

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

  PRINT_IF_ERROR(curandGenerateUniform(gen, c, M_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(curandDestroyGenerator(gen));

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  dim3 gridDim;
  dim3 blockDim;

  blockDim.x = BLOCK_ROW_TILES * WARP_SIZE;
  blockDim.y = BLOCK_COL_TILES;

  gridDim.x = (M_GLOBAL + (M * BLOCK_ROW_TILES - 1)) / (M * BLOCK_ROW_TILES);
  gridDim.y = (N_GLOBAL + N * blockDim.y - 1) / (N * blockDim.y);

  for (auto _ : state) {
    PRINT_IF_ERROR(cudaEventRecord(start));

    (compute_gemm_naive<<<gridDim, blockDim>>>(a_fp16, b_fp16, c, M_GLOBAL, N_GLOBAL,
                                               K_GLOBAL, alpha, beta));

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
                         {"num_elements", M * N * K},
                         {"flops",
                          {state.iterations() * 2.0 * M_GLOBAL * N_GLOBAL * K_GLOBAL,
                           benchmark::Counter::kAvgThreadsRate}}});

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  PRINT_IF_ERROR(cudaFree(a_fp32));
  PRINT_IF_ERROR(cudaFree(b_fp32));
  PRINT_IF_ERROR(cudaFree(a_fp16));
  PRINT_IF_ERROR(cudaFree(b_fp16));
  PRINT_IF_ERROR(cudaFree(c));

  cudaDeviceReset();
}

static void CUDA_WMMA_HGEMM_NAIVE(benchmark::State &state) {

  const auto M_GLOBAL = state.range(0);
  const auto N_GLOBAL = state.range(1);
  const auto K_GLOBAL = state.range(2);

  const __half alpha = approx_float_to_half(1.1f);
  const __half beta  = approx_float_to_half(1.2f);

  float *a_fp32;
  float *b_fp32;
  float *c_fp32;
  half *a_fp16;
  half *b_fp16;
  half *c_fp16;

  curandGenerator_t gen;

  // Use tensor cores
  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp32, M_GLOBAL * K_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp32, K_GLOBAL * N_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp16, M_GLOBAL * K_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp16, K_GLOBAL * N_GLOBAL * sizeof(half)));

  PRINT_IF_ERROR(cudaMalloc((void **) &c_fp32, M_GLOBAL * N_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &c_fp16, M_GLOBAL * N_GLOBAL * sizeof(half)));

  PRINT_IF_ERROR(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  PRINT_IF_ERROR(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));

  PRINT_IF_ERROR(curandGenerateUniform(gen, a_fp32, M_GLOBAL * K_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, b_fp32, K_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, c_fp32, K_GLOBAL * N_GLOBAL));

  // curand doesn't currently support fp16 so we generate in fp32 and convert to
  // fp16.
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(M_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(
      a_fp16, a_fp32, M_GLOBAL * K_GLOBAL)));
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(K_GLOBAL * N_GLOBAL + 255) / 256, 256>>>(
      b_fp16, b_fp32, K_GLOBAL * N_GLOBAL)));
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(M_GLOBAL * N_GLOBAL + 255) / 256, 256>>>(
      c_fp16, c_fp32, M_GLOBAL * N_GLOBAL)));

  PRINT_IF_ERROR(curandDestroyGenerator(gen));

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  dim3 gridDim;
  dim3 blockDim;

  blockDim.x = BLOCK_ROW_TILES * WARP_SIZE;
  blockDim.y = BLOCK_COL_TILES;

  gridDim.x = (M_GLOBAL + (M * BLOCK_ROW_TILES - 1)) / (M * BLOCK_ROW_TILES);
  gridDim.y = (N_GLOBAL + N * blockDim.y - 1) / (N * blockDim.y);

  for (auto _ : state) {
    PRINT_IF_ERROR(cudaEventRecord(start));

    (compute_hgemm_naive<<<gridDim, blockDim>>>(a_fp16, b_fp16, c_fp16, M_GLOBAL, N_GLOBAL,
                                               K_GLOBAL, alpha, beta));

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
                         {"num_elements", M * N * K},
                         {"flops",
                          {state.iterations() * 2.0 * M_GLOBAL * N_GLOBAL * K_GLOBAL,
                           benchmark::Counter::kAvgThreadsRate}}});

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  PRINT_IF_ERROR(cudaFree(a_fp32));
  PRINT_IF_ERROR(cudaFree(b_fp32));
  PRINT_IF_ERROR(cudaFree(a_fp16));
  PRINT_IF_ERROR(cudaFree(b_fp16));
  PRINT_IF_ERROR(cudaFree(c_fp32));
  PRINT_IF_ERROR(cudaFree(c_fp16));

  cudaDeviceReset();
}

BENCHMARK(CUDA_WMMA_GEMM_NAIVE)->ARGS()->UseManualTime();
BENCHMARK(CUDA_WMMA_HGEMM_NAIVE)->ARGS()->UseManualTime();
