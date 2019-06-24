#include <benchmark/benchmark.h>

#include "gemv/args.hpp"
#include "init/init.hpp"
#include "utils/utils.hpp"

#include <mma.h>
using namespace nvcuda;

#define NAIVE 1

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
static const int BLOCK_ROW_WARPS = 9;
static const int BLOCK_COL_WARPS = 1;

// number of WMMA tiles (16 X 16) processed by one warp
static const int WARP_ROW_TILES = 1;
static const int WARP_COL_TILES = 1;

// number of WMMA tiles for col and rwo in one block
static const int BLOCK_ROW_TILES = WARP_ROW_TILES * BLOCK_ROW_WARPS;
static const int BLOCK_COL_TILES = WARP_COL_TILES * BLOCK_COL_WARPS;

// number of warps and threads in one block
static const int WARPS_PER_BLOCK   = BLOCK_ROW_WARPS * BLOCK_COL_WARPS;
static const int THREADS_PER_BLOCK = WARP_SIZE * WARPS_PER_BLOCK;

// each block processes one tile at a time
static const int TILE_WIDTH_M = BLOCK_ROW_TILES * M;
static const int TILE_WIDTH_N = BLOCK_COL_TILES * N; // TILE_WIDTH_N <= TILE_WIDTH_M
static const int TILE_WIDTH_K = TILE_WIDTH_M;        // TILE_WIDTH_K <= TILE_WIDTH_M

static __global__ void changeValue(float *out, float *in, int n) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    out[idx] = in[idx];
  }
}

static __global__ void compute_wmma_gemv_naive(half *a, half *b, float *c, int M_GLOBAL,
                                               int N_GLOBAL, int K_GLOBAL, float alpha,
                                               float beta) {
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

static __global__ void compute_wmma_gemv_sharedmem(half *a, half *b, float *c,
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
      // wmma::load_matrix_sync(b_frag, (half *) subTileB + subtileBRow +
      // subtileBCol * TILE_WIDTH_K, TILE_WIDTH_K);

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

void doCUDA_WMMA_GEMV_ACCURACY(int M_GLOBAL, int K_GLOBAL) {

  const auto N_GLOBAL = BLOCK_COL_TILES * N;

  float alpha = 1.0f;
  float beta  = 0.0f;

  float *a_fp32;
  float *x_fp32;
  half *a_fp16;
  half *x_fp16;
  half *b_fp16;

  float *y;
  float *y_cublas;
  float *y_naive;

  float *y_host_cublas;
  float *y_host_naive;

  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp32, M_GLOBAL * K_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &x_fp32, K_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &a_fp16, M_GLOBAL * K_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &x_fp16, K_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &b_fp16, K_GLOBAL * N_GLOBAL * sizeof(half)));

  PRINT_IF_ERROR(cudaMalloc((void **) &y, M_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &y_cublas, M_GLOBAL * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc((void **) &y_naive, M_GLOBAL * sizeof(float)));

  y_host_cublas = (float *) malloc(M_GLOBAL * sizeof(float));
  y_host_naive  = (float *) malloc(M_GLOBAL * sizeof(float));

  curandGenerator_t gen;
  PRINT_IF_ERROR(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
  PRINT_IF_ERROR(curandSetPseudoRandomGeneratorSeed(gen, 1337ULL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, a_fp32, M_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, x_fp32, N_GLOBAL));
  PRINT_IF_ERROR(curandGenerateUniform(gen, y, M_GLOBAL));
  PRINT_IF_ERROR(curandDestroyGenerator(gen));

  PRINT_IF_LAUNCH_ERROR((changeValue<<<(M_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(
      a_fp32, a_fp32, M_GLOBAL * K_GLOBAL)));
  PRINT_IF_LAUNCH_ERROR(
      (changeValue<<<(K_GLOBAL + 255) / 256, 256>>>(x_fp32, x_fp32, K_GLOBAL)));

  // curand doesn't currently support fp16 so we generate in fp32 and convert to
  // fp16.
  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(M_GLOBAL * K_GLOBAL + 255) / 256, 256>>>(
      a_fp16, a_fp32, M_GLOBAL * K_GLOBAL)));
  PRINT_IF_LAUNCH_ERROR(
      (convertFp32ToFp16<<<(K_GLOBAL + 255) / 256, 256>>>(x_fp16, x_fp32, K_GLOBAL)));

  // copy vector x to matrix b, column-major
  PRINT_IF_ERROR(cudaMemset(b_fp16, 0, K_GLOBAL * N_GLOBAL * sizeof(half)));
  PRINT_IF_ERROR(
      cudaMemcpy(b_fp16, x_fp16, K_GLOBAL * sizeof(half), cudaMemcpyDeviceToDevice));

  PRINT_IF_ERROR(
      cudaMemcpy(y_cublas, y, M_GLOBAL * sizeof(float), cudaMemcpyDeviceToDevice));
  PRINT_IF_ERROR(
      cudaMemcpy(y_naive, y, M_GLOBAL * sizeof(float), cudaMemcpyDeviceToDevice));

  // First: using NAIVE
  dim3 gridDim;
  dim3 blockDim;

  blockDim.x = BLOCK_ROW_TILES * WARP_SIZE;
  blockDim.y = BLOCK_COL_TILES;

  gridDim.x = (M_GLOBAL + (M * BLOCK_ROW_TILES - 1)) / (M * BLOCK_ROW_TILES);
  gridDim.y = (N_GLOBAL + N * BLOCK_COL_TILES - 1) / (N * BLOCK_COL_TILES); // 1

#if NAIVE
  PRINT_IF_LAUNCH_ERROR((compute_wmma_gemv_naive<<<gridDim, blockDim>>>(
      a_fp16, b_fp16, y_naive, M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta)));
#else
  PRINT_IF_LAUNCH_ERROR((compute_wmma_gemv_sharedmem<<<gridDim, blockDim>>>(
      a_fp16, b_fp16, y_naive, M_GLOBAL, N_GLOBAL, K_GLOBAL, alpha, beta)));
#endif

  PRINT_IF_ERROR(cudaDeviceSynchronize());

  // Second: using CUBLAS
  cublasHandle_t cublasHandle;
  PRINT_IF_ERROR(cublasCreate(&cublasHandle));
  PRINT_IF_ERROR(cublasSetMathMode(cublasHandle, CUBLAS_DEFAULT_MATH));

  const int incx = 1;
  const int incy = 1;

  PRINT_IF_ERROR(cublasSgemv(cublasHandle, CUBLAS_OP_N, M_GLOBAL, K_GLOBAL, &alpha,
                             a_fp32, M_GLOBAL, x_fp32, incx, &beta, y_cublas, incy));

  PRINT_IF_ERROR(cudaMemcpy(y_host_naive, y_naive, M_GLOBAL * sizeof(float),
                            cudaMemcpyDeviceToHost));
  PRINT_IF_ERROR(cudaMemcpy(y_host_cublas, y_cublas, M_GLOBAL * sizeof(float),
                            cudaMemcpyDeviceToHost));

  // 0.01% relative tolerance. 1e-5 absolute tolerance.
  int errors = 0;
  for (int i = 0; i < M_GLOBAL; i++) {
    float v1 = y_host_cublas[i];
    float v2 = y_host_naive[i];
    if (v1 / v2 > 1.0001 || v2 / v1 > 1.0001 || abs(v1 - v2) > 1e-5) {
      errors++;
      /* printf("%f %f\n", v1, v2); */
    }
  }

  if (errors > 0) {
    printf("NAIVE does not agree with CUBLAS! %d errors!\n", errors);
  } else {
    printf("Results verified: they agree.\n\n");
  }
}

static void CUDA_WMMA_GEMV_ACCURACY(benchmark::State &state) {
  // M_GLOBAL, N_GLOBAL, K_GLOBAL must be multiple of M, N and K
  const auto M_GLOBAL = state.range(0);
  const auto K_GLOBAL = state.range(1);

  doCUDA_WMMA_GEMV_ACCURACY(M_GLOBAL, K_GLOBAL);
}
