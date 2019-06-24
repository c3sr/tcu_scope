
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
// number of warps needed for col and row in one block
static const int BLOCK_COL_WARPS = 1;
static const int BLOCK_ROW_WARPS = 9;

// number of WMMA tiles (16 X 16) processed by one warp
static const int WARP_COL_TILES = 1;
static const int WARP_ROW_TILES = 1;

// number of WMMA tiles for col and rwo in one block
static const int BLOCK_COL_TILES = WARP_COL_TILES * BLOCK_COL_WARPS;
static const int BLOCK_ROW_TILES = WARP_ROW_TILES * BLOCK_ROW_WARPS;

// number of warps and threads in one block
static const int WARPS_PER_BLOCK   = BLOCK_ROW_WARPS * BLOCK_COL_WARPS;
static const int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

// each block processes one tile at a time
static const int TILE_WIDTH_M = BLOCK_ROW_TILES * M;
static const int TILE_WIDTH_N = BLOCK_COL_TILES * N; // TILE_WIDTH_N <= TILE_WIDTH_M
static const int TILE_WIDTH_K = TILE_WIDTH_M;        // TILE_WIDTH_K <= TILE_WIDTH_M

static __global__ void compute_wmma_gemv_sharedmem(const half *__restrict__ a,
                                                   const half *__restrict__ b, float *c,
                                                   int M_GLOBAL, int N_GLOBAL,
                                                   int K_GLOBAL, float alpha,
                                                   float beta) {

  __shared__ half subTileA[TILE_WIDTH_K][TILE_WIDTH_M];
  __shared__ half subTileB[TILE_WIDTH_N][TILE_WIDTH_K];

  int tx = threadIdx.x;
  // int ty = 0; // threadIdx.y;
  int tid = tx; // threadIdx.y * blockDim.x + threadIdx.x; // thread id in the block

  int aRow = blockIdx.x * TILE_WIDTH_M; // staring row of the current block in matrix A
  // int bCol = 0; // blockIdx.y * TILE_WIDTH_N; // staring col of the current
  // block in matrix B

  // Declare the fragments
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::col_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, M, N, K, float> acc_frag;
  wmma::fragment<wmma::accumulator, M, N, K, float> c_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  for (int k = 0; k < K_GLOBAL; k += TILE_WIDTH_K) {
    // Collaborative loading of M tiles into shared memory
    for (int i = 0; i < TILE_WIDTH_M * TILE_WIDTH_K; i += THREADS_PER_BLOCK) {
      int idx = (tid + i);
      int aX  = idx % TILE_WIDTH_M;
      int aY  = idx / TILE_WIDTH_M;

      if (((k + aY) < K_GLOBAL) && ((aRow + aX) < M_GLOBAL)) {
        subTileA[aY][aX] = a[(k + aY) * M_GLOBAL + aRow + aX];
      } else {
        subTileA[aY][aX] = half(0);
      }
    }

    // Collaborative loading N tiles into shared memory
    for (int i = 0; i < TILE_WIDTH_K * TILE_WIDTH_N; i += THREADS_PER_BLOCK) {
      int idx = (tid + i);
      int bX  = idx % TILE_WIDTH_K;
      int bY  = idx / TILE_WIDTH_K;

      if ((bY < N_GLOBAL) && ((k + bX) < K_GLOBAL)) {
        subTileB[bY][bX] = b[bY * K_GLOBAL + k + bX];
        // subTileB[bY][bX] = (((bCol + bY) < N_GLOBAL) && ((k + bX) <
        // K_GLOBAL)) ? b[(bCol + bY) * K_GLOBAL + k + bX] : half(0);
      } else {
        subTileB[bY][bX] = half(0);
      }
    }

    __syncthreads();

    for (int i = 0; i < TILE_WIDTH_K; i += K) {
      int subtileARow = M * (threadIdx.x / WARP_SIZE);
      int subtileACol = i;

      int subtileBRow = i;
      // int subtileBCol = 0; // N * threadIdx.y;

      // Load the inputs
      wmma::load_matrix_sync(a_frag,
                             (half *) subTileA + subtileARow + subtileACol * TILE_WIDTH_M,
                             TILE_WIDTH_M);
      wmma::load_matrix_sync(b_frag, (half *) subTileB + subtileBRow, TILE_WIDTH_K);

      // Perform the matrix multiplication
      wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    }
  }

  // Load in the current value of c, scale it by beta, and add this our result
  // scaled by alpha
  int warpM = (blockIdx.x * blockDim.x + tx) / WARP_SIZE;
  // int warpN = 0; // blockIdx.y * blockDim.y + ty;
  int cRow = warpM * M;
  // int cCol  = 0; // warpN * N;

  if (cRow < M_GLOBAL) {
    wmma::load_matrix_sync(
        c_frag, c + cRow, M_GLOBAL,
        wmma::mem_col_major); // wmma::load_matrix_sync(c_frag, c + cRow + cCol
                              // * K_GLOBAL, M_GLOBAL, wmma::mem_col_major);

    for (int i = 0; i < c_frag.num_elements; i++) {
      c_frag.x[i] = alpha * acc_frag.x[i] + beta * c_frag.x[i];
    }
    // Store the output
    wmma::store_matrix_sync(c + cRow, c_frag, K_GLOBAL,
                            wmma::mem_col_major); // wmma::store_matrix_sync(c + cRow +
                                                  // cCol * K_GLOBAL, c_frag, K_GLOBAL,
                                                  // wmma::mem_col_major);
  }
}

static void CUDA_WMMA_GEMV_SHAREDMEM(benchmark::State &state) {
  const auto M_GLOBAL = state.range(0);
  const auto K_GLOBAL = state.range(1);
  const auto N_GLOBAL = BLOCK_COL_TILES * N;

  const float alpha = 1.1f;
  const float beta  = 1.2f;

  float *a_fp32;
  float *x_fp32;
  float *y;

  half *a_fp16;
  half *x_fp16;
  half *b_fp16;

  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp32, M_GLOBAL * K_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &x_fp32, K_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &y,
                            M_GLOBAL * N_GLOBAL *
                                sizeof(float))); // the first column holds the result

  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp16, M_GLOBAL * K_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &x_fp16, K_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp16, K_GLOBAL * N_GLOBAL * sizeof(half)));

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

    (compute_wmma_gemv_sharedmem<<<gridDim, blockDim>>>(a_fp16, b_fp16, y, M_GLOBAL,
                                                        N_GLOBAL, K_GLOBAL, alpha, beta));

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

// BENCHMARK(CUDA_WMMA_GEMV_SHAREDMEM)->ARGS()->UseManualTime();
