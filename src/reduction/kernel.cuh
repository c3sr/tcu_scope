#pragma once

#include "init/init.hpp"
#include "utils/utils.hpp"

#include <cooperative_groups.h>
#include <mma.h>

namespace wmma_reduction {

using namespace nvcuda;

using namespace cooperative_groups;

#ifndef WARP_SIZE
#define WARP_SIZE (32)
#endif // WARP_SIZE

// MMA matrix tile dimensions. (16, 16, 16), (32, 8, 16), and (8, 32,
// 16) are currently supported.
static const int M              = 16;
static const int N              = 16;
static const int K              = 16;
static const int WMMA_TILE_SIZE = (M * N);

static constexpr __host__ __device__ int min_unroll(size_t x, size_t y) {
  return x <= y ? x : y;
}

// segment_size = 16
// each warp calculates WMMA_TILES_PER_WARP * 16 segments
template <int WMMA_TILES_PER_WARP, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_reduction_16(const half *__restrict__ d_in,
                                                           half *__restrict__ d_out,
                                                           size_t num_segments) {

  __shared__ half r_frag_s[WMMA_TILE_SIZE];
  __shared__ half d_out_s[WARPS_PER_BLOCK * WMMA_TILES_PER_WARP * WMMA_TILE_SIZE];

  const size_t globalWarpIdx = (blockIdx.x * BLOCK_DIM + threadIdx.x) / WARP_SIZE;
  const int localWarpIdx     = threadIdx.x / WARP_SIZE;
  const int laneid           = threadIdx.x % WARP_SIZE;

#pragma unroll
  for (int idx = threadIdx.x; idx < WMMA_TILE_SIZE; idx += BLOCK_DIM) {
    const auto ii = idx % N;
    r_frag_s[idx] = ii == 0 ? one<half>() : zero<half>();
  }

  __syncthreads();

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  wmma::load_matrix_sync(r_frag, r_frag_s, 16);

#pragma unroll
  for (int ii = 0; ii < WMMA_TILES_PER_WARP; ii++) {
    const size_t globalTileIdx    = globalWarpIdx * WMMA_TILES_PER_WARP + ii;
    const size_t globalSegmentIdx = globalTileIdx * 16;
    const size_t offset           = globalTileIdx * WMMA_TILE_SIZE;
    const int d_out_s_offset = (localWarpIdx * WMMA_TILES_PER_WARP + ii) * WMMA_TILE_SIZE;

    wmma::fill_fragment(out_frag, zero<half>());
    wmma::load_matrix_sync(a_frag, d_in + offset, 16);

    wmma::mma_sync(out_frag, a_frag, r_frag, out_frag);

    wmma::store_matrix_sync(d_out_s + d_out_s_offset, out_frag, 16, wmma::mem_col_major);

    // copy the strided results from d_out_s to d_out
    if (laneid < 16) {
      d_out[globalSegmentIdx + laneid] = d_out_s[d_out_s_offset + laneid];
    }
  }
}

// each warp calculates 16 consecutive segments
template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_reduction_16n(const half *__restrict__ d_in,
                                                            half *__restrict__ d_out,
                                                            size_t num_segments) {

  __shared__ half r_frag_s[WMMA_TILE_SIZE];
  __shared__ half d_out_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];

  const size_t globalWarpIdx    = (blockIdx.x * BLOCK_DIM + threadIdx.x) / WARP_SIZE;
  const size_t globalSegmentIdx = globalWarpIdx * 16;
  const size_t global_offset    = globalSegmentIdx * SEGMENT_SIZE;
  const int localWarpIdx        = threadIdx.x / WARP_SIZE;
  const int local_offset        = localWarpIdx * WMMA_TILE_SIZE;
  const int laneid              = threadIdx.x % WARP_SIZE;

#pragma unroll
  for (int idx = threadIdx.x; idx < WMMA_TILE_SIZE; idx += BLOCK_DIM) {
    const auto ii = idx % N;
    r_frag_s[idx] = ii == 0 ? one<half>() : zero<half>();
  }

  __syncthreads();

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  wmma::load_matrix_sync(r_frag, r_frag_s, 16);
  wmma::fill_fragment(out_frag, zero<half>());

#pragma unroll min_unroll(SEGMENT_SIZE / 16, 32)
  for (size_t ii = 0; ii < SEGMENT_SIZE / 16; ii++) {
    const size_t offset = global_offset + ii * 16;

    wmma::load_matrix_sync(a_frag, d_in + offset, SEGMENT_SIZE);

    wmma::mma_sync(out_frag, a_frag, r_frag, out_frag);
  }

  wmma::store_matrix_sync(d_out_s + local_offset, out_frag, 16, wmma::mem_col_major);

  // copy the strided results from d_out_s to d_out
  if (laneid < 16) {
    d_out[globalSegmentIdx + laneid] = d_out_s[local_offset + laneid];
  }
}

// each warp calculates strided 16 segments
template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_reduction_16n_opt(
    const half *__restrict__ d_in, half *__restrict__ d_out, size_t num_segments) {

  __shared__ half r_frag_s[WMMA_TILE_SIZE];
  __shared__ half d_out_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];

  const int localWarpIdx        = threadIdx.x / WARP_SIZE;
  const int local_offset        = localWarpIdx * WMMA_TILE_SIZE;
  const int laneid              = threadIdx.x % WARP_SIZE;
  const size_t globalSegmentIdx = blockIdx.x * WARPS_PER_BLOCK + localWarpIdx;
  const size_t global_offset    = globalSegmentIdx * SEGMENT_SIZE;

#pragma unroll
  for (int idx = threadIdx.x; idx < WMMA_TILE_SIZE; idx += BLOCK_DIM) {
    const auto ii = idx % N;
    r_frag_s[idx] = ii == 0 ? one<half>() : zero<half>();
  }

  __syncthreads();

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  wmma::load_matrix_sync(r_frag, r_frag_s, 16);
  wmma::fill_fragment(out_frag, zero<half>());

#pragma unroll min_unroll(SEGMENT_SIZE / 16, 32)
  for (size_t ii = 0; ii < SEGMENT_SIZE / 16; ii++) {
    const size_t offset = global_offset + ii * 16;

    // WARPS_PER_BLOCK * SEGMENT_SIZE cannot be more than 2^31 - 1 = 2147483647
    wmma::load_matrix_sync(a_frag, d_in + offset, WARPS_PER_BLOCK * SEGMENT_SIZE);

    wmma::mma_sync(out_frag, a_frag, r_frag, out_frag);
  }

  wmma::store_matrix_sync(d_out_s + local_offset, out_frag, 16, wmma::mem_col_major);

  // copy the strided results from d_out_s to d_out
  if (laneid < 16) {
    d_out[globalSegmentIdx + laneid * WARPS_PER_BLOCK] = d_out_s[local_offset + laneid];
  }
}

// each block calculates consecutive 16 segments
template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_reduction_16n_block(
    const half *__restrict__ d_in, half *__restrict__ d_out, size_t num_segments) {

  __shared__ half r_frag_s[WMMA_TILE_SIZE];
  __shared__ half d_out_s[16 * WMMA_TILE_SIZE];

  const size_t wmma_tiles_per_segment = SEGMENT_SIZE / 16;
  const int localWarpIdx              = threadIdx.x / WARP_SIZE;
  const int local_offset              = localWarpIdx * WMMA_TILE_SIZE;
  const int laneid                    = threadIdx.x % WARP_SIZE;

  const size_t global_offset = blockIdx.x * 16 * SEGMENT_SIZE;

#pragma unroll
  for (int ii = 0;; ii += BLOCK_DIM) {
    const auto idx = (threadIdx.x + ii);
    const auto col = idx % N;
    if (idx >= WMMA_TILE_SIZE) {
      break;
    }
    r_frag_s[idx] = col == 0 ? one<half>() : zero<half>();
#pragma unroll
    for (int jj = 0; jj < 16; jj++) {
      d_out_s[idx + jj * WMMA_TILE_SIZE] = 0;
    }
  }

  __syncthreads();

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::col_major> r_t_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  wmma::load_matrix_sync(r_frag, r_frag_s, 16);
  wmma::load_matrix_sync(r_t_frag, r_frag_s, 16);
  wmma::fill_fragment(out_frag, zero<half>());

#pragma unroll
  for (size_t ii = 0; ii < wmma_tiles_per_segment; ii += WARPS_PER_BLOCK) {
    const size_t offset = global_offset + (localWarpIdx + ii) * 16;

    wmma::load_matrix_sync(a_frag, d_in + offset, SEGMENT_SIZE);

    wmma::mma_sync(out_frag, a_frag, r_frag, out_frag);
  }

  wmma::store_matrix_sync(d_out_s + local_offset, out_frag, 16, wmma::mem_col_major);

  if (localWarpIdx == 0) {
    wmma::fill_fragment(out_frag, zero<half>());
    wmma::load_matrix_sync(b_frag, d_out_s, 256);
    wmma::mma_sync(out_frag, r_t_frag, b_frag, out_frag);
    wmma::store_matrix_sync(d_out_s, out_frag, 16, wmma::mem_row_major);

    if (laneid < 16) {
      d_out[blockIdx.x * 16 + laneid] = d_out_s[laneid];
    }
  }
}

// segment_size = WMMA_TILE_SIZE
// each warp calculates SEGMENTS_PER_WARP segments
template <int SEGMENTS_PER_WARP, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_reduction_256(const half *__restrict__ d_in,
                                                            half *__restrict__ d_out,
                                                            size_t num_segments) {

  __shared__ half ra_mat_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];
  __shared__ half r_frag_s[WMMA_TILE_SIZE];
  __shared__ half d_out_s[WARPS_PER_BLOCK * SEGMENTS_PER_WARP * WMMA_TILE_SIZE];

  const size_t globalWarpIdx = (blockIdx.x * BLOCK_DIM + threadIdx.x) / WARP_SIZE;
  const int localWarpIdx     = threadIdx.x / WARP_SIZE;
  const int local_offset     = localWarpIdx * WMMA_TILE_SIZE;
  const int laneid           = threadIdx.x % WARP_SIZE;

#pragma unroll
  for (int idx = threadIdx.x; idx < WMMA_TILE_SIZE; idx += BLOCK_DIM) {
    const auto ii = idx / N;
    r_frag_s[idx] = ii == 0 ? one<half>() : zero<half>();
  }

  __syncthreads();

  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> r_t_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> ra_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> ra_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  wmma::load_matrix_sync(r_frag, r_frag_s, 16);
  wmma::load_matrix_sync(r_t_frag, r_frag_s, 16);

#pragma unroll
  for (int ii = 0; ii < SEGMENTS_PER_WARP; ii++) {
    const size_t globalSegmentIdx = globalWarpIdx * SEGMENTS_PER_WARP + ii;
    const size_t offset           = globalSegmentIdx * WMMA_TILE_SIZE;
    const int d_out_s_offset = (localWarpIdx * SEGMENTS_PER_WARP + ii) * WMMA_TILE_SIZE;

    wmma::fill_fragment(ra_frag, zero<half>());
    wmma::fill_fragment(out_frag, zero<half>());
    wmma::load_matrix_sync(a_frag, d_in + offset, 16);

    wmma::mma_sync(ra_frag, r_frag, a_frag, ra_frag);

    // store accumulator ra_frag into shared memory and load it into
    // matrix_a fragment ra_mat_frag
    wmma::store_matrix_sync(ra_mat_s + local_offset, ra_frag, 16, wmma::mem_row_major);
    wmma::load_matrix_sync(ra_mat_frag, ra_mat_s + local_offset, 16);

    wmma::mma_sync(out_frag, ra_mat_frag, r_t_frag, out_frag);

    wmma::store_matrix_sync(d_out_s + d_out_s_offset, out_frag, 16, wmma::mem_row_major);

    // copy the strided results from d_out_s to d_out
    if (laneid == 0) {
      d_out[globalSegmentIdx] = d_out_s[d_out_s_offset];
    }
  }
}

// each warp calculates 1 segment
template <typename OUT_TYPE, size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_reduction_256n(
    const half *__restrict__ d_in, OUT_TYPE *__restrict__ d_out, size_t num_segments) {

  __shared__ half r_frag_s[WMMA_TILE_SIZE];
  __shared__ OUT_TYPE ra_mat_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];
  __shared__ OUT_TYPE d_out_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];

  const size_t globalWarpIdx  = (blockIdx.x * BLOCK_DIM + threadIdx.x) / WARP_SIZE;
  const size_t global_offset  = globalWarpIdx * SEGMENT_SIZE;
  const int localWarpIdx      = threadIdx.x / WARP_SIZE;
  const int local_offset      = localWarpIdx * WMMA_TILE_SIZE;
  const int laneid            = threadIdx.x % WARP_SIZE;
  const size_t num_wmma_tiles = (SEGMENT_SIZE + WMMA_TILE_SIZE - 1) / WMMA_TILE_SIZE;

#pragma unroll
  for (int idx = threadIdx.x; idx < WMMA_TILE_SIZE; idx += BLOCK_DIM) {
    const auto ii = idx / N;
    r_frag_s[idx] = ii == 0 ? one<half>() : zero<half>();
  }

  __syncthreads();

  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> r_t_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> ra_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> ra_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, OUT_TYPE> out_frag;

  wmma::load_matrix_sync(r_frag, r_frag_s, 16);
  wmma::load_matrix_sync(r_t_frag, r_frag_s, 16);
  wmma::fill_fragment(out_frag, zero<OUT_TYPE>());
  wmma::fill_fragment(ra_frag, zero<half>());

#pragma unroll
  for (size_t ii = 0; ii < num_wmma_tiles; ii++) {
    const size_t offset = global_offset + ii * WMMA_TILE_SIZE;
    wmma::load_matrix_sync(a_frag, d_in + offset, 16);
    wmma::mma_sync(ra_frag, r_frag, a_frag, ra_frag);
  }

  // store accumulator ra_frag into shared memory and load it into
  // matrix_a fragment ra_mat_frag
  wmma::store_matrix_sync(ra_mat_s + local_offset, ra_frag, 16, wmma::mem_row_major);
  wmma::load_matrix_sync(ra_mat_frag, ra_mat_s + local_offset, 16);

  wmma::mma_sync(out_frag, ra_mat_frag, r_t_frag, out_frag);

  wmma::store_matrix_sync(d_out_s + local_offset, out_frag, 16, wmma::mem_row_major);

  // copy the strided results from d_out_s to d_out
  if (laneid == 0) {
    d_out[globalWarpIdx] = d_out_s[local_offset];
  }
}

// each warp calculates 1 segment
template <typename OUT_TYPE, size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_reduction_256n_org(
    const half *__restrict__ d_in, OUT_TYPE *__restrict__ d_out, size_t num_segments) {

  __shared__ half r_frag_s[WMMA_TILE_SIZE];
  __shared__ half ra_mat_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];
  __shared__ OUT_TYPE d_out_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];

  const size_t globalWarpIdx  = (blockIdx.x * BLOCK_DIM + threadIdx.x) / WARP_SIZE;
  const int localWarpIdx      = threadIdx.x / WARP_SIZE;
  const int local_offset      = localWarpIdx * WMMA_TILE_SIZE;
  const int laneid            = threadIdx.x % WARP_SIZE;
  const size_t num_wmma_tiles = (SEGMENT_SIZE + WMMA_TILE_SIZE - 1) / WMMA_TILE_SIZE;

#pragma unroll
  for (int idx = threadIdx.x; idx < WMMA_TILE_SIZE; idx += BLOCK_DIM) {
    const auto ii = idx / N;
    r_frag_s[idx] = ii == 0 ? one<half>() : zero<half>();
  }

  __syncthreads();

  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> r_t_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> ra_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> ra_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, OUT_TYPE> out_frag;

  wmma::load_matrix_sync(r_frag, r_frag_s, 16);
  wmma::load_matrix_sync(r_t_frag, r_frag_s, 16);
  wmma::fill_fragment(out_frag, zero<OUT_TYPE>());

#pragma unroll
  for (size_t ii = 0; ii < num_wmma_tiles; ii++) {
    const size_t global_offset = globalWarpIdx * SEGMENT_SIZE + ii * WMMA_TILE_SIZE;

    wmma::fill_fragment(ra_frag, zero<half>());
    wmma::load_matrix_sync(a_frag, d_in + global_offset, 16);

    wmma::mma_sync(ra_frag, r_frag, a_frag, ra_frag);

    // store accumulator ra_frag into shared memory and load it into
    // matrix_a fragment ra_mat_frag
    wmma::store_matrix_sync(ra_mat_s + local_offset, ra_frag, 16, wmma::mem_row_major);
    wmma::load_matrix_sync(ra_mat_frag, ra_mat_s + local_offset, 16);

    wmma::mma_sync(out_frag, ra_mat_frag, r_t_frag, out_frag);
  }

  wmma::store_matrix_sync(d_out_s + local_offset, out_frag, 16, wmma::mem_row_major);

  // copy the strided results from d_out_s to d_out
  if (laneid == 0) {
    d_out[globalWarpIdx] = d_out_s[local_offset];
  }
}

// each block calculates 1 segment
template <typename OUT_TYPE, size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_reduction_256n_block(
    const half *__restrict__ d_in, OUT_TYPE *__restrict__ d_out, size_t num_segments) {

  __shared__ half r_frag_s[WMMA_TILE_SIZE];
  __shared__ OUT_TYPE ra_mat_s[16 * WMMA_TILE_SIZE];
  __shared__ OUT_TYPE d_out_s[WMMA_TILE_SIZE];

  const size_t globalSegmentIdx       = blockIdx.x;
  const size_t wmma_tiles_per_segment = SEGMENT_SIZE / WMMA_TILE_SIZE;
  const size_t global_offset          = globalSegmentIdx * SEGMENT_SIZE;
  const int localWarpIdx              = threadIdx.x / WARP_SIZE;
  const int local_offset              = localWarpIdx * WMMA_TILE_SIZE;
  const int laneid                    = threadIdx.x % WARP_SIZE;

#pragma unroll
  for (int ii = 0;; ii += BLOCK_DIM) {
    const auto idx = (threadIdx.x + ii);
    const auto row = idx / N;
    if (idx >= WMMA_TILE_SIZE) {
      break;
    }
    r_frag_s[idx] = row == 0 ? one<half>() : zero<half>();
#pragma unroll
    for (int jj = 0; jj < 16; jj++) {
      ra_mat_s[idx + jj * WMMA_TILE_SIZE] = 0;
    }
  }

  __syncthreads();

  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> r_t_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> ra_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> ra_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, OUT_TYPE> out_frag;

  wmma::load_matrix_sync(r_frag, r_frag_s, 16);
  wmma::load_matrix_sync(r_t_frag, r_frag_s, 16);
  wmma::fill_fragment(out_frag, zero<OUT_TYPE>());
  wmma::fill_fragment(ra_frag, zero<half>());

#pragma unroll
  for (size_t ii = 0; ii < wmma_tiles_per_segment; ii += WARPS_PER_BLOCK) {
    const size_t offset = global_offset + (ii + localWarpIdx) * WMMA_TILE_SIZE;
    wmma::load_matrix_sync(a_frag, d_in + offset, 16);
    wmma::mma_sync(ra_frag, r_frag, a_frag, ra_frag);
  }

  wmma::store_matrix_sync(ra_mat_s + local_offset, ra_frag, 16, wmma::mem_row_major);

  if (localWarpIdx == 0) {
    wmma::load_matrix_sync(ra_mat_frag, ra_mat_s, 256);
    wmma::mma_sync(out_frag, ra_mat_frag, r_t_frag, out_frag);

    wmma::store_matrix_sync(d_out_s, out_frag, 16, wmma::mem_row_major);

    half val = laneid < 16 ? d_out_s[laneid] : zero<half>();
#pragma unroll
    for (int offset = 16 / 2; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (threadIdx.x == 0) {
      d_out[globalSegmentIdx] = d_out_s[0];
    }
  }
}

// each block calculates 1 segment
template <typename OUT_TYPE, size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_reduction_256n_block_org(
    const half *__restrict__ d_in, OUT_TYPE *__restrict__ d_out, size_t num_segments) {

  __shared__ half r_frag_s[WMMA_TILE_SIZE];
  __shared__ OUT_TYPE ra_mat_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];
  __shared__ OUT_TYPE d_out_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];

  const size_t globalSegmentIdx       = blockIdx.x;
  const size_t wmma_tiles_per_segment = SEGMENT_SIZE / WMMA_TILE_SIZE;
  const size_t global_offset          = globalSegmentIdx * SEGMENT_SIZE;
  const int localWarpIdx              = threadIdx.x / WARP_SIZE;
  const int laneid                    = threadIdx.x % WARP_SIZE;
  const int local_offset              = localWarpIdx * WMMA_TILE_SIZE;

#pragma unroll
  for (int ii = 0;; ii += BLOCK_DIM) {
    const auto idx = (threadIdx.x + ii);
    const auto row = idx / N;
    if (idx >= WMMA_TILE_SIZE) {
      break;
    }
    r_frag_s[idx] = row == 0 ? one<half>() : zero<half>();
  }

  __syncthreads();

  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> r_t_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> ra_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> ra_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, OUT_TYPE> out_frag;

  wmma::load_matrix_sync(r_frag, r_frag_s, 16);
  wmma::load_matrix_sync(r_t_frag, r_frag_s, 16);
  wmma::fill_fragment(out_frag, zero<OUT_TYPE>());
  wmma::fill_fragment(ra_frag, zero<half>());

#pragma unroll
  for (size_t ii = 0; ii < wmma_tiles_per_segment; ii += WARPS_PER_BLOCK) {
    const size_t offset = global_offset + (ii + localWarpIdx) * WMMA_TILE_SIZE;
    wmma::load_matrix_sync(a_frag, d_in + offset, 16);
    wmma::mma_sync(ra_frag, r_frag, a_frag, ra_frag);
  }

  // store accumulator ra_frag into shared memory and load it into
  // matrix_a fragment ra_mat_frag
  wmma::store_matrix_sync(ra_mat_s + local_offset, ra_frag, 16, wmma::mem_row_major);
  wmma::load_matrix_sync(ra_mat_frag, ra_mat_s + local_offset, 16);

  wmma::mma_sync(out_frag, ra_mat_frag, r_t_frag, out_frag);

  wmma::store_matrix_sync(d_out_s + local_offset, out_frag, 16, wmma::mem_row_major);

  if (laneid == 0) {
    d_out_s[localWarpIdx] = d_out_s[local_offset];
  }

  __syncthreads();

  if (localWarpIdx == 0) {
    half val = laneid < WARPS_PER_BLOCK ? d_out_s[laneid] : zero<half>();
#pragma unroll
    for (int offset = WARPS_PER_BLOCK / 2; offset > 0; offset >>= 1) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (threadIdx.x == 0) {
      d_out[globalSegmentIdx] = d_out_s[0];
    }
  }
}

// SEGMENTS_PER_WARP = 1
// each warp calculates 1 segment
template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_reduction_cg(const half *__restrict__ d_in,
                                                 half *__restrict__ d_out,
                                                 size_t num_segments) {

  __shared__ half r_frag_s[WMMA_TILE_SIZE];
  __shared__ half ra_mat_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];
  __shared__ half d_out_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];

  const size_t globalThreadIdx = blockIdx.x * BLOCK_DIM + threadIdx.x;
  const size_t globalWarpIdx   = globalThreadIdx / WARP_SIZE;
  const int localWarpIdx       = threadIdx.x / WARP_SIZE;
  const size_t global_offset   = globalWarpIdx * SEGMENT_SIZE;
  const int local_offset       = localWarpIdx * WMMA_TILE_SIZE;
  const size_t num_wmma_tiles  = (SEGMENT_SIZE + WMMA_TILE_SIZE - 1) / WMMA_TILE_SIZE;

#pragma unroll
  for (int idx = threadIdx.x; idx < WMMA_TILE_SIZE; idx += BLOCK_DIM) {
    const auto ii = idx / N;
    r_frag_s[idx] = ii == 0 ? one<half>() : zero<half>();
  }

  __syncthreads();

  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> r_t_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> ra_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> ra_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  wmma::load_matrix_sync(r_frag, r_frag_s, 16);
  wmma::load_matrix_sync(r_t_frag, r_frag_s, 16);
  wmma::fill_fragment(out_frag, zero<half>());
  wmma::fill_fragment(ra_frag, zero<half>());

#pragma unroll
  for (size_t ii = 0; ii < num_wmma_tiles; ii++) {
    const size_t offset = global_offset + ii * WMMA_TILE_SIZE;

    wmma::load_matrix_sync(a_frag, d_in + offset, 16);

    wmma::mma_sync(ra_frag, r_frag, a_frag, ra_frag);
  }

  // store accumulator ra_frag into shared memory and load it into
  // matrix_a fragment ra_mat_frag
  wmma::store_matrix_sync(ra_mat_s + local_offset, ra_frag, 16, wmma::mem_row_major);
  wmma::load_matrix_sync(ra_mat_frag, ra_mat_s + local_offset, 16);

  wmma::mma_sync(out_frag, ra_mat_frag, r_t_frag, out_frag);

  wmma::store_matrix_sync(d_out_s + local_offset, out_frag, 16, wmma::mem_row_major);

  __syncthreads();

  if (threadIdx.x == 0) {
    half block_accum = zero<half>();
#pragma unroll
    for (int ii = 0; ii < WARPS_PER_BLOCK; ii++) {
      block_accum += d_out_s[ii * WMMA_TILE_SIZE];
    }
    d_out[blockIdx.x] = block_accum;
  }

  this_grid().sync();

  if (globalThreadIdx == 0) {
    half global_accum = zero<half>();
#pragma unroll
    for (int ii = 0; ii < gridDim.x; ii++) {
      global_accum += d_out[ii];
    }
    d_out[0] = global_accum;
  }
}

// SEGMENTS_PER_WARP = 1
// each warp calculates 1 segment
template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_reduction_atomic_w_syncthreads(
    const half *__restrict__ d_in, half *__restrict__ d_out, size_t num_segments) {

  __shared__ half block_reduction_value[8]; // to maintain alignment
  __shared__ half r_frag_s[WMMA_TILE_SIZE];
  __shared__ half ra_mat_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];
  __shared__ half d_out_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];

  const size_t globalThreadIdx = blockIdx.x * BLOCK_DIM + threadIdx.x;
  const size_t globalWarpIdx   = globalThreadIdx / WARP_SIZE;
  const int localWarpIdx       = threadIdx.x / WARP_SIZE;
  const size_t global_offset   = globalWarpIdx * SEGMENT_SIZE;
  const int local_offset       = localWarpIdx * WMMA_TILE_SIZE;
  const int laneid             = threadIdx.x % WARP_SIZE;
  const size_t num_wmma_tiles  = (SEGMENT_SIZE + WMMA_TILE_SIZE - 1) / WMMA_TILE_SIZE;

#pragma unroll
  for (int idx = threadIdx.x; idx < WMMA_TILE_SIZE; idx += BLOCK_DIM) {
    const auto ii = idx / N;
    r_frag_s[idx] = ii == 0 ? one<half>() : zero<half>();
  }

  if (threadIdx.x == 0) {
    block_reduction_value[0] = 0;
  }

  __syncthreads();

  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> r_t_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> ra_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> ra_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  wmma::load_matrix_sync(r_frag, r_frag_s, 16);
  wmma::load_matrix_sync(r_t_frag, r_frag_s, 16);
  wmma::fill_fragment(out_frag, zero<half>());
  wmma::fill_fragment(ra_frag, zero<half>());

#pragma unroll
  for (size_t ii = 0; ii < num_wmma_tiles; ii++) {
    const size_t offset = global_offset + ii * WMMA_TILE_SIZE;

    wmma::load_matrix_sync(a_frag, d_in + offset, 16);

    wmma::mma_sync(ra_frag, r_frag, a_frag, ra_frag);
  }

  // store accumulator ra_frag into shared memory and load it into
  // matrix_a fragment ra_mat_frag
  wmma::store_matrix_sync(ra_mat_s + local_offset, ra_frag, 16, wmma::mem_row_major);
  wmma::load_matrix_sync(ra_mat_frag, ra_mat_s + local_offset, 16);

  wmma::mma_sync(out_frag, ra_mat_frag, r_t_frag, out_frag);

  wmma::store_matrix_sync(d_out_s + local_offset, out_frag, 16, wmma::mem_row_major);

  // local reduction
  if (laneid == 0) {
    xprintf("d_out_s[%d] = %f -- d_out[0] = %f\n", local_offset,
            float(d_out_s[local_offset]), float(d_out[0]));
    atomicAdd(&block_reduction_value[0], d_out_s[local_offset]);
  }
  __syncthreads();
  // global reduction
  if (threadIdx.x == 0) {
    atomicAdd(&d_out[0], block_reduction_value[0]);
    xprintf("block_reduction = %f -- d_out[0] = %f\n", float(block_reduction_value[0]),
            float(d_out[0]));
  }
}

// SEGMENTS_PER_WARP = 1
// each warp calculates 1 segment
template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_reduction_atomic_w_atomicballot(
    const half *__restrict__ d_in, half *__restrict__ d_out, size_t num_segments) {

  __shared__ unsigned int block_reduction_visitor;
  __shared__ half block_reduction_value[6]; // to maintain alignment
  __shared__ half r_frag_s[WMMA_TILE_SIZE];
  __shared__ half ra_mat_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];
  __shared__ half d_out_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];

  const size_t globalThreadIdx = blockIdx.x * BLOCK_DIM + threadIdx.x;
  const size_t globalWarpIdx   = globalThreadIdx / WARP_SIZE;
  const int localWarpIdx       = threadIdx.x / WARP_SIZE;
  const size_t global_offset   = globalWarpIdx * SEGMENT_SIZE;
  const int local_offset       = localWarpIdx * WMMA_TILE_SIZE;
  const int laneid             = threadIdx.x % WARP_SIZE;
  const size_t num_warps       = BLOCK_DIM / WARP_SIZE;
  const size_t num_wmma_tiles  = (SEGMENT_SIZE + WMMA_TILE_SIZE - 1) / WMMA_TILE_SIZE;

#pragma unroll
  for (int idx = threadIdx.x; idx < WMMA_TILE_SIZE; idx += BLOCK_DIM) {
    const auto ii = idx / N;
    r_frag_s[idx] = ii == 0 ? one<half>() : zero<half>();
  }

  if (threadIdx.x == 0) {
    block_reduction_value[0] = 0;
    block_reduction_visitor  = 0;
  }

  __syncthreads();

  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> r_t_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> ra_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> ra_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  wmma::load_matrix_sync(r_frag, r_frag_s, 16);
  wmma::load_matrix_sync(r_t_frag, r_frag_s, 16);
  wmma::fill_fragment(out_frag, zero<half>());
  wmma::fill_fragment(ra_frag, zero<half>());

#pragma unroll
  for (size_t ii = 0; ii < num_wmma_tiles; ii++) {
    const size_t offset = global_offset + ii * WMMA_TILE_SIZE;

    wmma::load_matrix_sync(a_frag, d_in + offset, 16);

    wmma::mma_sync(ra_frag, r_frag, a_frag, ra_frag);
  }

  // store accumulator ra_frag into shared memory and load it into
  // matrix_a fragment ra_mat_frag
  wmma::store_matrix_sync(ra_mat_s + local_offset, ra_frag, 16, wmma::mem_row_major);
  wmma::load_matrix_sync(ra_mat_frag, ra_mat_s + local_offset, 16);

  wmma::mma_sync(out_frag, ra_mat_frag, r_t_frag, out_frag);

  wmma::store_matrix_sync(d_out_s + local_offset, out_frag, 16, wmma::mem_row_major);

  if (laneid == 0) {
    atomicAdd(&block_reduction_value[0], d_out_s[local_offset]);
    if (atomicInc(&block_reduction_visitor, num_warps) == (num_warps - 1)) {
      atomicAdd(&d_out[0], block_reduction_value[0]);
    }
  }
}
} // namespace wmma_reduction
