/*
 * Copyright 1993-2017 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

// CUDA sample demonstrating a GEMM computation using the Warp Matrix Multiply
// and Accumulate API introduced in CUDA 9.

// In this program, the compute_gemm kernel computes the result of a matrix
// multiplication and addition: D = alpha * A * B + beta * C. The dimensions of
// both C and D matrices are M_GLOBAL x N_GLOBAL. The A matrix is M_GLOBAL x
// K_GLOBAL (row-major), the B matrix is K_GLOBAL x N_GLOBAL (column-major). In
// that kernel, each CTA computes one 128 x 128 tile of the resulting matrix per
// iteration. When the tile is computed, the CTA stores it to the global memory
// and begins a new iteration, selecting a new 128 x 128 tile to compute.
// Each CTA consists of eight warps. For the 128 x 128 tile, each warp computes
// eight 16 x 16 subtiles, organized in a 2 x 4 two-dimensional array. Warps
// compute the 16 x 16 subtiles using nvcuda::wmma::mma_sync operations by
// moving through the K_GLOBAL dimension of the A and B matrices and
// accumulating the intermediate result in the local thread state.

// There are a number of simple optimizations used in the algorithm:
// - The CTA copies the 128 x 128 tile of the C matrix from the global memory to
//   shared memory. After that is done, each warp loads the C matrix fragments
//   from shared memory, thus avoiding a random global memory access.
// - On each internal iteration, the CTA copies a portion of the A and B
// matrices from
//   global memory to shared memory. After that, all warps in the CTA reuse the
//   A and B data from shared memory, thus reducing the number of data copies
//   from global memory.
// - The portions of the A and B matrices are stored in shared memory with an
// additional
//   padding (skew) to reduce the number of shared memory access bank conflicts.
//   (See a detailed explanation near the SKEW_HALF macro definition.)
// - When the CTA finishes computing the tiles of the resulting matrix, each
// warp stores
//   its subtiles to shared memory. The CTA then copies the shared memory
//   contents to global memory, again avoiding redundant random global memory
//   accesses.
// - Note that the CTA tile size is chosen to maximize the GPU register
// utilization,
//   but carefully enough to avoid local memory use.

// cuda-9.1_samples/0_Simple/cudaTensorCoreGemm

#include <benchmark/benchmark.h>

#include "gemm/args.hpp"
#include "init/init.hpp"
#include "utils/utils.hpp"

#include <assert.h>
#include <mma.h>
#include <stdio.h>

#ifndef WARP_SIZE
#define WARP_SIZE (32)
#endif // WARP_SIZE

// MMA matrix tile dimensions. (16, 16, 16), (32, 8, 16), and (8, 32, 16) are
// currently supported.
static const int M = 16;
static const int N = 16;
static const int K = 16;

// Implementation constants.
static const int WARPS_PER_BLOCK   = 8;
static const int THREADS_PER_BLOCK = (WARP_SIZE * WARPS_PER_BLOCK);

static const int CHUNK_K = 8;

static const int BLOCK_ROW_WARPS = 2;
static const int BLOCK_COL_WARPS = 4;

static const int WARP_ROW_TILES = 4;
static const int WARP_COL_TILES = 2;

static const int BLOCK_ROW_TILES = (WARP_ROW_TILES * BLOCK_ROW_WARPS);
static const int BLOCK_COL_TILES = (WARP_COL_TILES * BLOCK_COL_WARPS);

static const int SHMEM_STRIDE = (N * BLOCK_ROW_TILES);
static const int SHMEM_OFFSET = (N * WARP_ROW_TILES);

#define C_LAYOUT wmma::mem_row_major

// The macro below is used to shift rows of the A matrix and columns of the B
// matrix in shared memory to minimize possible bank conflicts. Before
// performing the nvcuda::wmma::mma_sync operation, the warp must load the
// matrix data using the nvcuda::wmma::load_matrix_sync operation. Although the
// memory access pattern is not specified for that function, each lane in the
// warp can read one or multiple matrix elements from different matrix rows or
// columns. For shared memory, such access can result in bank conflicts if
// different rows / columns of the matrix map to the same bank. By shifting each
// row and column by a few bytes, we make sure that they map to different banks,
// thus reducing the number of possible bank conflicts. The number of 8 two-byte
// "half" elements is chosen as the minimum possible shift because we must keep
// each row and column 128-bit aligned, as required by
// nvcuda::wmma::load_matrix_sync.

static const int SKEW_HALF = 8;

using namespace nvcuda;

__host__ void init_host_matrices(float *a, float *b, float *c, int M_GLOBAL, int N_GLOBAL,
                                 int K_GLOBAL) {
  for (int i = 0; i < M_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      a[i * K_GLOBAL + j] = (float) (rand() % 3);
    }
  }

  for (int i = 0; i < N_GLOBAL; i++) {
    for (int j = 0; j < K_GLOBAL; j++) {
      b[i * K_GLOBAL + j] = (float) (rand() % 3);
    }
  }

  for (int t = 0; t < M_GLOBAL * N_GLOBAL; t++) {
    c[t] = (float) (rand() % 3);
  }
}

__global__ void init_gemm_device_matrices(const float *A_h, const float *B_h,
                                          const float *C_h, half *A, half *B, float *C,
                                          float *D, int M_GLOBAL, int N_GLOBAL,
                                          int K_GLOBAL) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M_GLOBAL * K_GLOBAL;
       i += gridDim.x * blockDim.x)
    A[i] = __float2half(A_h[i]);

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < N_GLOBAL * K_GLOBAL;
       i += gridDim.x * blockDim.x)
    B[i] = __float2half(B_h[i]);

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M_GLOBAL * N_GLOBAL;
       i += gridDim.x * blockDim.x)
    C[i] = C_h[i];

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M_GLOBAL * N_GLOBAL;
       i += gridDim.x * blockDim.x)
    D[i] = 0;
}

__global__ void init_hgemm_device_matrices(const float *A_h, const float *B_h,
                                           const float *C_h, half *A, half *B, half *C,
                                           half *D, int M_GLOBAL, int N_GLOBAL,
                                           int K_GLOBAL) {
  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M_GLOBAL * K_GLOBAL;
       i += gridDim.x * blockDim.x)
    A[i] = __float2half(A_h[i]);

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < N_GLOBAL * K_GLOBAL;
       i += gridDim.x * blockDim.x)
    B[i] = __float2half(B_h[i]);

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M_GLOBAL * N_GLOBAL;
       i += gridDim.x * blockDim.x)
    C[i] = __float2half(C_h[i]);

  for (int i = blockDim.x * blockIdx.x + threadIdx.x; i < M_GLOBAL * N_GLOBAL;
       i += gridDim.x * blockDim.x)
    D[i] = zero<half>();
}

__global__ void compute_gemm(const half *A, const half *B, const float *C, float *D,
                             float alpha, float beta, int M_GLOBAL, int N_GLOBAL,
                             int K_GLOBAL) {
  extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

  const auto M_TILES = M_GLOBAL / M;
  const auto N_TILES = N_GLOBAL / N;
  const auto K_TILES = K_GLOBAL / K;

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  // Offset in shared memory from which the B matrix is stored.
  const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

  // This pointer is used to access the C and D matrix tiles this warp computes.
  float *shmem_warp_tile_ptr = (float *) &shmem[0][0] +
                               (warpId / 2) * SHMEM_STRIDE * K * 2 +
                               (warpId % 2) * SHMEM_OFFSET;

  // This pointer is used to stream the C and D matrices block-wide tile to and
  // from shared memory.
  float *shmem_warp_stream_ptr = (float *) &shmem[0][0] + warpId * SHMEM_STRIDE * K;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may
  // result in a loss of precision). Zero still needs to be specially handled
  // though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the
  // matrix to the right and down, and selects the next tile to compute. Once
  // there's no such tile, all warps in this CTA exit.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_tile_i =
        ((block_pos * BLOCK_COL_TILES) / N_TILES) * (BLOCK_ROW_WARPS * WARP_ROW_TILES);
    const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) {
      break;
    }

    // This warp's pointer to the C matrix data to copy memory from to shared
    // memory.
    const size_t gmem_idx = (block_tile_i + warpId) * M * N_GLOBAL + block_tile_j * N;
    const float *src_gmem_warp_stream_ptr = &C[gmem_idx];

// Stream multiple C tiles to shared memory.
#pragma unroll
    for (int i = 0; i < K; i++) {
      typedef int4 copy_t;

      *((copy_t *) (shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
          *((copy_t *) (src_gmem_warp_stream_ptr + N_GLOBAL * i) + laneId);
    }

    __syncthreads();

    // These fragments will accumulate the result of A and B matrix fragment
    // multiplications along the K_GLOBAL dimension.
    wmma::fragment<wmma::accumulator, M, N, K, float> c[WARP_COL_TILES][WARP_ROW_TILES];

// Load the C matrix tiles into fragments from shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        const float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

// Scale the C matrix.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        for (int t = 0; t < c[i][j].num_elements; t++) {
          c[i][j].x[t] *= beta;
        }
      }
    }

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
    const half *warp_ptr =
        (warpId < 4)
            ? (&A[block_tile_i * M * K_GLOBAL] + M * K_GLOBAL * (warpId % 4) * 2)
            : (&B[block_tile_j * N * K_GLOBAL] + N * K_GLOBAL * (warpId % 4) * 2);

// Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest copy
      // the B matrix.
      size_t shmem_idx =
          warpId < (WARPS_PER_BLOCK / 2)
              ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
              : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      int4 *lane_ptr =
          (int4 *) (warp_ptr + tile_k * K + (laneId / (WARP_SIZE / 2)) * K_GLOBAL) +
          (laneId % (WARP_SIZE / 2));

      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      shmem_idx += laneId / (WARP_SIZE / 2);

#pragma unroll
      for (int i = 0; i < (WARP_SIZE / 2); i++) {
        // Copy 16 bytes at once in each lane.
        *((int4 *) &shmem[shmem_idx][0] + (laneId % (WARP_SIZE / 2))) = *lane_ptr;

        // Advance the global memory pointer and the shared memory index.
        lane_ptr = (int4 *) ((half *) lane_ptr + K_GLOBAL * 2);
        shmem_idx += 2;
      }

      __syncthreads();

// Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a   = (warpId / 2) * M * 2 + (i * M);
          const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

          wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b =
                  shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId % 2) + (j * N);
              const half *tile_ptr = &shmem[shmem_idx_b][k_step * K];

              wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
            }

            wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
          }
        }
      }

      __syncthreads();
    }

// Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        // Uniform, point-wise transformations of ALL fragment elements by ALL
        // threads in the warp are well-defined even though element indices
        // within fragment storage are not defined.
        for (int t = 0; t < c[i][j].num_elements; t++)
          c[i][j].x[t] *= alpha;

        float *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // Now that shared memory contains all the D tiles, stream them to global
    // memory.
    float *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
    for (int i = 0; i < K; i++) {
      *((int4 *) (dst_gmem_warp_stream_ptr + N_GLOBAL * i) + laneId) =
          *((int4 *) (shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
    }

    __syncthreads();
  }
}

__global__ void compute_hgemm(const half *A, const half *B, const half *C, half *D,
                              half alpha, half beta, int M_GLOBAL, int N_GLOBAL,
                              int K_GLOBAL) {
  extern __shared__ half shmem[][CHUNK_K * K + SKEW_HALF];

  const auto M_TILES = M_GLOBAL / M;
  const auto N_TILES = N_GLOBAL / N;
  const auto K_TILES = K_GLOBAL / K;

  // Warp and lane identification.
  const unsigned int warpId = threadIdx.x / WARP_SIZE;
  const unsigned int laneId = threadIdx.x % WARP_SIZE;

  // Offset in shared memory from which the B matrix is stored.
  const size_t shmem_idx_b_off = BLOCK_COL_TILES * M;

  // This pointer is used to access the C and D matrix tiles this warp computes.
  half *shmem_warp_tile_ptr = (half *) &shmem[0][0] +
                              (warpId / 2) * SHMEM_STRIDE * K * 2 +
                              (warpId % 2) * SHMEM_OFFSET;

  // This pointer is used to stream the C and D matrices block-wide tile to and
  // from shared memory.
  half *shmem_warp_stream_ptr = (half *) &shmem[0][0] + warpId * SHMEM_STRIDE * K;

  // Adjust the beta scaler, as it'll be multiplied by alpha at the end of
  // each tile computation. Technically this is not generally correct (may
  // result in a loss of precision). Zero still needs to be specially handled
  // though.
  beta /= alpha;

  // Each CTA slides along the 128 x 128 tiles from the top left corner of the
  // matrix to the right and down, and selects the next tile to compute. Once
  // there's no such tile, all warps in this CTA exit.
  for (unsigned int block_pos = blockIdx.x;; block_pos += gridDim.x) {
    const unsigned int block_tile_i =
        ((block_pos * BLOCK_COL_TILES) / N_TILES) * (BLOCK_ROW_WARPS * WARP_ROW_TILES);
    const unsigned int block_tile_j = (block_pos * BLOCK_COL_TILES) % N_TILES;

    // Stop when there are no more D matrix tiles to compute in this CTA.
    if (block_tile_i >= M_TILES) {
      break;
    }

    // This warp's pointer to the C matrix data to copy memory from to shared
    // memory.
    const size_t gmem_idx = (block_tile_i + warpId) * M * N_GLOBAL + block_tile_j * N;
    const half *src_gmem_warp_stream_ptr = &C[gmem_idx];

// Stream multiple C tiles to shared memory.
#pragma unroll
    for (int i = 0; i < K; i++) {
      typedef int2 copy_t;

      *((copy_t *) (shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId) =
          *((copy_t *) (src_gmem_warp_stream_ptr + N_GLOBAL * i) + laneId);
    }

    __syncthreads();

    // These fragments will accumulate the result of A and B matrix fragment
    // multiplications along the K_GLOBAL dimension.
    wmma::fragment<wmma::accumulator, M, N, K, half> c[WARP_COL_TILES][WARP_ROW_TILES];

// Load the C matrix tiles into fragments from shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
        const half *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        wmma::load_matrix_sync(c[i][j], tile_ptr, SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

// Scale the C matrix.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        for (int t = 0; t < c[i][j].num_elements; t++) {
          c[i][j].x[t] *= beta;
        }
      }
    }

    // Select what warp copies what matrix to shared memory.
    // Warps 0-3 copy the A matrix, warps 4-7 copy the B matrix.
    const half *warp_ptr =
        (warpId < 4)
            ? (&A[block_tile_i * M * K_GLOBAL] + M * K_GLOBAL * (warpId % 4) * 2)
            : (&B[block_tile_j * N * K_GLOBAL] + N * K_GLOBAL * (warpId % 4) * 2);

// Go through the global K dimension by a fixed step at a time.
#pragma unroll
    for (int tile_k = 0; tile_k < K_TILES; tile_k += CHUNK_K) {
      // Copy slices of the A and B matrices to shared memory.
      // The first half of the warps in the CTA copy the A matrix, the rest copy
      // the B matrix.
      size_t shmem_idx =
          warpId < (WARPS_PER_BLOCK / 2)
              ? (M * (warpId % (WARPS_PER_BLOCK / 2)) * 2)
              : (N * (warpId % (WARPS_PER_BLOCK / 2)) * 2 + shmem_idx_b_off);

      // First half of the warp copies the first row / column of the matrix,
      // the second half of the warp copies the next.
      int2 *lane_ptr =
          (int2 *) (warp_ptr + tile_k * K + (laneId / (WARP_SIZE / 2)) * K_GLOBAL) +
          (laneId % (WARP_SIZE / 2));

      // Shift the second half of the warp to the next row / column in the
      // shared memory.
      shmem_idx += laneId / (WARP_SIZE / 2);

#pragma unroll
      for (int i = 0; i < (WARP_SIZE / 2); i++) {
        // Copy 16 bytes at once in each lane.
        *((int2 *) &shmem[shmem_idx][0] + (laneId % (WARP_SIZE / 2))) = *lane_ptr;

        // Advance the global memory pointer and the shared memory index.
        lane_ptr = (int2 *) ((half *) lane_ptr + K_GLOBAL * 2);
        shmem_idx += 2;
      }

      __syncthreads();

// Compute a grid of C matrix tiles in each warp.
#pragma unroll
      for (int k_step = 0; k_step < CHUNK_K; k_step++) {
        wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a[WARP_COL_TILES];
        wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> b[WARP_ROW_TILES];

#pragma unroll
        for (int i = 0; i < WARP_COL_TILES; i++) {
          size_t shmem_idx_a   = (warpId / 2) * M * 2 + (i * M);
          const half *tile_ptr = &shmem[shmem_idx_a][k_step * K];

          wmma::load_matrix_sync(a[i], tile_ptr, K * CHUNK_K + SKEW_HALF);

#pragma unroll
          for (int j = 0; j < WARP_ROW_TILES; j++) {
            if (i == 0) {
              // Load the B matrix fragment once, because it is going to be
              // reused against the other A matrix fragments.
              size_t shmem_idx_b =
                  shmem_idx_b_off + (WARP_ROW_TILES * N) * (warpId % 2) + (j * N);
              const half *tile_ptr = &shmem[shmem_idx_b][k_step * K];

              wmma::load_matrix_sync(b[j], tile_ptr, K * CHUNK_K + SKEW_HALF);
            }

            wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
          }
        }
      }

      __syncthreads();
    }

// Store the D fragments to shared memory.
#pragma unroll
    for (int i = 0; i < WARP_COL_TILES; i++) {
#pragma unroll
      for (int j = 0; j < WARP_ROW_TILES; j++) {
#pragma unroll
        // Uniform, point-wise transformations of ALL fragment elements by ALL
        // threads in the warp are well-defined even though element indices
        // within fragment storage are not defined.
        for (int t = 0; t < c[i][j].num_elements; t++)
          c[i][j].x[t] *= alpha;

        half *tile_ptr = shmem_warp_tile_ptr + i * SHMEM_STRIDE * K + j * N;

        wmma::store_matrix_sync(tile_ptr, c[i][j], SHMEM_STRIDE, C_LAYOUT);
      }
    }

    __syncthreads();

    // Now that shared memory contains all the D tiles, stream them to global
    // memory.
    half *dst_gmem_warp_stream_ptr = &D[gmem_idx];

#pragma unroll
    for (int i = 0; i < K; i++) {
      *((int2 *) (dst_gmem_warp_stream_ptr + N_GLOBAL * i) + laneId) =
          *((int2 *) (shmem_warp_stream_ptr + SHMEM_STRIDE * i) + laneId);
    }

    __syncthreads();
  }
}

static void CUDA_WMMA_GEMM(benchmark::State &state) {
  /* if (!has_cuda) { */
  /*   state.SkipWithError(fmt::format("CUDA_WMMA_GEMM no CUDA device found"));
   */
  /*   return; */
  /* } */

  // M_GLOBAL, N_GLOBAL, K_GLOBAL must be multiple of M, N and K
  const auto M_GLOBAL = state.range(0);
  const auto N_GLOBAL = state.range(1);
  const auto K_GLOBAL = state.range(2);

  const float alpha = 1.1f;
  const float beta  = 1.2f;

  int dev = 0;
  cudaDeviceProp deviceProp;
  PRINT_IF_ERROR(cudaGetDeviceProperties(&deviceProp, dev));

  // Tensor cores require a GPU of Volta (SM7X) architecture or higher.
  if (deviceProp.major < 7) {
    printf("cudaTensorCoreGemm requires requires SM 7.0 or higher to use "
           "Tensor Cores.  Exiting...\n");
    exit(-1);
  }

  float *A_h = NULL;
  float *B_h = NULL;
  float *C_h = NULL;

  PRINT_IF_ERROR(cudaMallocManaged((void **) &A_h, sizeof(float) * M_GLOBAL * K_GLOBAL));
  PRINT_IF_ERROR(cudaMallocManaged((void **) &B_h, sizeof(float) * K_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(cudaMallocManaged((void **) &C_h, sizeof(float) * M_GLOBAL * N_GLOBAL));

  half *A  = NULL;
  half *B  = NULL;
  float *C = NULL;
  float *D = NULL;

  PRINT_IF_ERROR(cudaMalloc((void **) &A, sizeof(half) * M_GLOBAL * K_GLOBAL));
  PRINT_IF_ERROR(cudaMalloc((void **) &B, sizeof(half) * N_GLOBAL * K_GLOBAL));
  PRINT_IF_ERROR(cudaMalloc((void **) &C, sizeof(float) * M_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(cudaMalloc((void **) &D, sizeof(float) * M_GLOBAL * N_GLOBAL));

  assert(((unsigned long long) A) % 128 == 0);
  assert(((unsigned long long) B) % 128 == 0);
  assert(((unsigned long long) C) % 128 == 0);
  assert(((unsigned long long) D) % 128 == 0);

  init_host_matrices(A_h, B_h, C_h, M_GLOBAL, N_GLOBAL, K_GLOBAL);

  PRINT_IF_LAUNCH_ERROR(
      (init_gemm_device_matrices<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK>>>(
          A_h, B_h, C_h, A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL)));

  PRINT_IF_ERROR(cudaDeviceSynchronize());

  enum {
    SHMEM_SZ = sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2
  };

  // printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

  PRINT_IF_ERROR(cudaFuncSetAttribute(
      compute_gemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    PRINT_IF_ERROR(cudaEventRecord(start));

    (compute_gemm<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ>>>(
        A, B, C, D, alpha, beta, M_GLOBAL, N_GLOBAL, K_GLOBAL));

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

  PRINT_IF_ERROR(cudaEventDestroy(start));
  PRINT_IF_ERROR(cudaEventDestroy(stop));

  PRINT_IF_ERROR(cudaFree((void *) A_h));
  PRINT_IF_ERROR(cudaFree((void *) B_h));
  PRINT_IF_ERROR(cudaFree((void *) C_h));
  PRINT_IF_ERROR(cudaFree((void *) A));
  PRINT_IF_ERROR(cudaFree((void *) B));
  PRINT_IF_ERROR(cudaFree((void *) C));
  PRINT_IF_ERROR(cudaFree((void *) D));

  cudaDeviceReset();
}

static void CUDA_WMMA_HGEMM(benchmark::State &state) {

  // M_GLOBAL, N_GLOBAL, K_GLOBAL must be multiple of M, N and K
  const auto M_GLOBAL = state.range(0);
  const auto N_GLOBAL = state.range(1);
  const auto K_GLOBAL = state.range(2);

  const __half alpha = approx_float_to_half(1.1f);
  const __half beta  = approx_float_to_half(1.2f);

  int dev = 0;
  cudaDeviceProp deviceProp;
  PRINT_IF_ERROR(cudaGetDeviceProperties(&deviceProp, dev));

  // Tensor cores require a GPU of Volta (SM7X) architecture or higher.
  if (deviceProp.major < 7) {
    printf("cudaTensorCoreGemm requires requires SM 7.0 or higher to use "
           "Tensor Cores.  Exiting...\n");
    exit(-1);
  }

  float *A_h = NULL;
  float *B_h = NULL;
  float *C_h = NULL;

  PRINT_IF_ERROR(cudaMallocManaged((void **) &A_h, sizeof(float) * M_GLOBAL * K_GLOBAL));
  PRINT_IF_ERROR(cudaMallocManaged((void **) &B_h, sizeof(float) * K_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(cudaMallocManaged((void **) &C_h, sizeof(float) * M_GLOBAL * N_GLOBAL));

  half *A = NULL;
  half *B = NULL;
  half *C = NULL;
  half *D = NULL;

  PRINT_IF_ERROR(cudaMalloc((void **) &A, sizeof(half) * M_GLOBAL * K_GLOBAL));
  PRINT_IF_ERROR(cudaMalloc((void **) &B, sizeof(half) * N_GLOBAL * K_GLOBAL));
  PRINT_IF_ERROR(cudaMalloc((void **) &C, sizeof(half) * M_GLOBAL * N_GLOBAL));
  PRINT_IF_ERROR(cudaMalloc((void **) &D, sizeof(half) * M_GLOBAL * N_GLOBAL));

  assert(((unsigned long long) A) % 128 == 0);
  assert(((unsigned long long) B) % 128 == 0);
  assert(((unsigned long long) C) % 128 == 0);
  assert(((unsigned long long) D) % 128 == 0);

  init_host_matrices(A_h, B_h, C_h, M_GLOBAL, N_GLOBAL, K_GLOBAL);

  PRINT_IF_LAUNCH_ERROR(
      (init_hgemm_device_matrices<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK>>>(
          A_h, B_h, C_h, A, B, C, D, M_GLOBAL, N_GLOBAL, K_GLOBAL)));

  PRINT_IF_ERROR(cudaDeviceSynchronize());

  enum {
    SHMEM_SZ = sizeof(half) * (BLOCK_COL_TILES * M) * (CHUNK_K * K + SKEW_HALF) * 2
  };

  // printf("Required shared memory size: %lu Kb\n", SHMEM_SZ / 1024UL);

  PRINT_IF_ERROR(cudaFuncSetAttribute(
      compute_hgemm, cudaFuncAttributeMaxDynamicSharedMemorySize, SHMEM_SZ));

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    PRINT_IF_ERROR(cudaEventRecord(start));

    (compute_hgemm<<<deviceProp.multiProcessorCount, THREADS_PER_BLOCK, SHMEM_SZ>>>(
        A, B, C, D, alpha, beta, M_GLOBAL, N_GLOBAL, K_GLOBAL));

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

  PRINT_IF_ERROR(cudaEventDestroy(start));
  PRINT_IF_ERROR(cudaEventDestroy(stop));

  PRINT_IF_ERROR(cudaFree((void *) A_h));
  PRINT_IF_ERROR(cudaFree((void *) B_h));
  PRINT_IF_ERROR(cudaFree((void *) C_h));
  PRINT_IF_ERROR(cudaFree((void *) A));
  PRINT_IF_ERROR(cudaFree((void *) B));
  PRINT_IF_ERROR(cudaFree((void *) C));
  PRINT_IF_ERROR(cudaFree((void *) D));

  cudaDeviceReset();
}

BENCHMARK(CUDA_WMMA_GEMM)->ARGS()->UseManualTime();
BENCHMARK(CUDA_WMMA_HGEMM)->ARGS()->UseManualTime();
