
#include <benchmark/benchmark.h>

#include "gemv/args.hpp"
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
static const int BLOCK_ROW_TILES = 16;
static const int BLOCK_COL_TILES = 1;

// Performs an GEMV y = alpha * Ax + beta * y assuming:
//  1) Matrices are packed in memory.
//  2) M and N are multiples of 16.
//  3) A is not transposed.
static __global__ void compute_wmma_gemv_naive(const half *__restrict__ a,
                                               const half *__restrict__ b, float *c,
                                               int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
                                               float alpha, float beta) {
  // Leading dimensions. Packed with no transpositions.
  int lda = M_GLOBAL;
  int ldb = K_GLOBAL;
  int ldc = M_GLOBAL;

  // Global warp id, warpN is 0.
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  // int warpN = 0; // (blockIdx.y * blockDim.y + threadIdx.y);

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, M, N, K, float> acc_frag;
  wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;

  wmma::fill_fragment(acc_frag, 0.0f);

  // Loop over k
  for (int i = 0; i < K_GLOBAL; i += K) {
    int aRow = warpM * M;
    int aCol = i;

    int bRow = i;
    // int bCol = 0; // warpN * N;

    // Bounds checking
    if (aRow < M_GLOBAL) { // if (aRow < M_GLOBAL && bCol < N_GLOBAL) {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
      wmma::load_matrix_sync(
          b_frag, b + bRow,
          ldb); // wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Load in the current value of c, scale it by beta, and add this our result
  // scaled by alpha
  int cRow = warpM * M;
  // int cCol = 0; // warpN * N;

  // printf("crow = %d ldc = %d warpM = %d M = %d\n", cRow, ldc, warpM, M);

  if (cRow < M_GLOBAL) { // if (cRow < M && cCol < N) {
    wmma::load_matrix_sync(
        c_frag, c + cRow, ldc,
        wmma::mem_col_major); // wmma::load_matrix_sync(c_frag, c + cRow + cCol
                              // * ldc, ldc, wmma::mem_col_major);

    for (int i = 0; i < c_frag.num_elements; i++) {
      c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }

    // Store the output
    wmma::store_matrix_sync(
        c + cRow, c_frag, ldc,
        wmma::mem_col_major); // wmma::store_matrix_sync(c + cRow + cCol * ldc,
                              // c_frag, ldc, wmma::mem_col_major);
  }
}

static __global__ void compute_wmma_hgemv_naive(const half *__restrict__ a,
                                                const half *__restrict__ b, half *c,
                                                int M_GLOBAL, int N_GLOBAL, int K_GLOBAL,
                                                half alpha, half beta) {
  // Leading dimensions. Packed with no transpositions.
  int lda = M_GLOBAL;
  int ldb = K_GLOBAL;
  int ldc = M_GLOBAL;

  // Global warp id, warpN is 0.
  int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
  // int warpN = 0; // (blockIdx.y * blockDim.y + threadIdx.y);

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
    // int bCol = 0; // warpN * N;

    // Bounds checking
    if (aRow < M_GLOBAL) { // if (aRow < M_GLOBAL && bCol < N_GLOBAL) {
      // Load the inputs
      wmma::load_matrix_sync(a_frag, a + aRow + aCol * lda, lda);
      wmma::load_matrix_sync(
          b_frag, b + bRow,
          ldb); // wmma::load_matrix_sync(b_frag, b + bRow + bCol * ldb, ldb);

      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Load in the current value of c, scale it by beta, and add this our result
  // scaled by alpha
  int cRow = warpM * M;
  // int cCol = 0; // warpN * N;

  // printf("crow = %d ldc = %d warpM = %d M = %d\n", cRow, ldc, warpM, M);

  if (cRow < M_GLOBAL) { // if (cRow < M && cCol < N) {
    wmma::load_matrix_sync(
        c_frag, c + cRow, ldc,
        wmma::mem_col_major); // wmma::load_matrix_sync(c_frag, c + cRow + cCol
                              // * ldc, ldc, wmma::mem_col_major);

    for (int i = 0; i < c_frag.num_elements; i++) {
      c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }

    // Store the output
    wmma::store_matrix_sync(
        c + cRow, c_frag, ldc,
        wmma::mem_col_major); // wmma::store_matrix_sync(c + cRow + cCol * ldc,
                              // c_frag, ldc, wmma::mem_col_major);
  }
}

static void CUDA_WMMA_GEMV_NAIVE(benchmark::State &state) {
  const auto M_GLOBAL = state.range(0);
  const auto K_GLOBAL = state.range(1);
  const auto N_GLOBAL = BLOCK_COL_TILES * N;

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
  PRINT_IF_ERROR(curandGenerateUniform(gen, y, M_GLOBAL));

  PRINT_IF_ERROR(curandDestroyGenerator(gen));

  // curand doesn't currently support fp16 so we generate in fp32 and convert to
  // fp16.
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(M_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(
      a_fp16, a_fp32, M_GLOBAL * K_GLOBAL)));
  PRINT_IF_LAUNCH_ERROR(
      (convertFp32ToFp16<<<(N_GLOBAL + 255) / 256, 256>>>(x_fp16, x_fp32, K_GLOBAL)));

  // copy vector x to matrix b, column-major
  PRINT_IF_ERROR(
      cudaMemcpy(b_fp16, x_fp16, K_GLOBAL * sizeof(half), cudaMemcpyDeviceToDevice));

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  dim3 gridDim;
  dim3 blockDim;

  blockDim.x = BLOCK_ROW_TILES * WARP_SIZE;
  blockDim.y = BLOCK_COL_TILES;

  gridDim.x = (M_GLOBAL + (M * BLOCK_ROW_TILES - 1)) / (M * BLOCK_ROW_TILES);
  gridDim.y = (N_GLOBAL + N * BLOCK_COL_TILES - 1) / (N * BLOCK_COL_TILES); // 1

  for (auto _ : state) {
    PRINT_IF_ERROR(cudaEventRecord(start));

    (compute_wmma_gemv_naive<<<gridDim, blockDim>>>(a_fp16, b_fp16, y, M_GLOBAL, N_GLOBAL,
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
                         {"N", K_GLOBAL},
                         {"num_elements", M_GLOBAL * K_GLOBAL},
                         {"flops",
                          {state.iterations() * 2.0 * M_GLOBAL * K_GLOBAL,
                           benchmark::Counter::kAvgThreadsRate}}});

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  PRINT_IF_ERROR(cudaFree(a_fp32));
  PRINT_IF_ERROR(cudaFree(x_fp32));
  PRINT_IF_ERROR(cudaFree(y));
  PRINT_IF_ERROR(cudaFree(a_fp16));
  PRINT_IF_ERROR(cudaFree(x_fp16));
  PRINT_IF_ERROR(cudaFree(b_fp16));

  cudaDeviceReset();
}

static void CUDA_WMMA_HGEMV_NAIVE(benchmark::State &state) {
  const auto M_GLOBAL = state.range(0);
  const auto K_GLOBAL = state.range(1);
  const auto N_GLOBAL = BLOCK_COL_TILES * N;

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

  dim3 gridDim;
  dim3 blockDim;

  blockDim.x = BLOCK_ROW_TILES * WARP_SIZE;
  blockDim.y = BLOCK_COL_TILES;

  gridDim.x = (M_GLOBAL + (M * BLOCK_ROW_TILES - 1)) / (M * BLOCK_ROW_TILES);
  gridDim.y = (N_GLOBAL + N * BLOCK_COL_TILES - 1) / (N * BLOCK_COL_TILES); // 1

  for (auto _ : state) {
    PRINT_IF_ERROR(cudaEventRecord(start));

    (compute_wmma_hgemv_naive<<<gridDim, blockDim>>>(a_fp16, b_fp16, y_fp16, M_GLOBAL,
                                                    N_GLOBAL, K_GLOBAL, alpha, beta));

    PRINT_IF_ERROR(cudaEventRecord(stop));
    PRINT_IF_ERROR(cudaEventSynchronize(stop));

    state.PauseTiming();

    float msecTotal = 0.0f;
    PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
    state.SetIterationTime(msecTotal / 1000);
    state.ResumeTiming();
  }

  state.counters.insert({{"M", M_GLOBAL},
                         {"N", K_GLOBAL},
                         {"num_elements", M_GLOBAL * K_GLOBAL},
                         {"flops",
                          {state.iterations() * 2.0 * M_GLOBAL * K_GLOBAL,
                           benchmark::Counter::kAvgThreadsRate}}});

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  PRINT_IF_ERROR(cudaFree(a_fp32));
  PRINT_IF_ERROR(cudaFree(x_fp32));
  PRINT_IF_ERROR(cudaFree(y_fp32));
  PRINT_IF_ERROR(cudaFree(y_fp16));
  PRINT_IF_ERROR(cudaFree(a_fp16));
  PRINT_IF_ERROR(cudaFree(x_fp16));
  PRINT_IF_ERROR(cudaFree(b_fp16));

  cudaDeviceReset();
}

BENCHMARK(CUDA_WMMA_GEMV_NAIVE)->ARGS()->UseManualTime();
BENCHMARK(CUDA_WMMA_HGEMV_NAIVE)->ARGS()->UseManualTime();
