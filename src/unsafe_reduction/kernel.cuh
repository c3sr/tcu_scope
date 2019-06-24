#pragma once

#include "init/init.hpp"
#include "utils/utils.hpp"

#include <cooperative_groups.h>
#include <mma.h>

namespace wmma_unsafe_reduction {

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

  const size_t globalWarpIdx = p2_div<WARP_SIZE>(blockIdx.x * BLOCK_DIM + threadIdx.x);
  const int localWarpIdx     = p2_div<WARP_SIZE>(threadIdx.x);
  const int laneid           = p2_mod<WARP_SIZE>(threadIdx.x);

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_b::set_first_row_ones(r_frag);

#pragma unroll
  for (int ii = 0; ii < WMMA_TILES_PER_WARP; ii++) {
    const size_t globalTileIdx    = globalWarpIdx * WMMA_TILES_PER_WARP + ii;
    const size_t globalSegmentIdx = p2_mul<16L>(globalTileIdx);
    const size_t offset           = p2_mul<WMMA_TILE_SIZE>(globalTileIdx);

    wmma::fill_fragment(out_frag, zero<half>());
    wmma::load_matrix_sync(a_frag, d_in + offset, 16);

    wmma::mma_sync(out_frag, a_frag, r_frag, out_frag);

    utils::fragment::matrix_c::store_first_row(&d_out[globalSegmentIdx], out_frag);
  }
}

// each warp calculates 16 consecutive segments
template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_reduction_16n(const half *__restrict__ d_in,
                                                            half *__restrict__ d_out,
                                                            size_t num_segments) {

  const size_t globalWarpIdx    = (blockIdx.x * BLOCK_DIM + threadIdx.x) / WARP_SIZE;
  const size_t globalSegmentIdx = p2_mul<16L>(globalWarpIdx);
  const size_t global_offset    = globalSegmentIdx * SEGMENT_SIZE;
  const int localWarpIdx        = p2_div<WARP_SIZE>(threadIdx.x);
  const int local_offset        = p2_mul<WMMA_TILE_SIZE>(localWarpIdx);
  const int laneid              = p2_mod<WARP_SIZE>(threadIdx.x);

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

#if 0
  utils::fragment::matrix_b::set_first_row_ones(r_frag);
#else // TODO: This is incorrect but requires fix to set_first_row_ones
  wmma::fill_fragment(r_frag, one<half>());
#endif

  wmma::fill_fragment(out_frag, zero<half>());

#pragma unroll
  for (size_t ii = 0; ii < SEGMENT_SIZE / 16; ii++) {
    const size_t offset = global_offset + ii * 16;

    wmma::load_matrix_sync(a_frag, d_in + offset, SEGMENT_SIZE);

    wmma::mma_sync(out_frag, a_frag, r_frag, out_frag);
  }

  utils::fragment::matrix_c::store_first_row(&d_out[globalSegmentIdx], out_frag);
}

// each warp calculates strided 16 segments
template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_reduction_16n_opt(
    const half *__restrict__ d_in, half *__restrict__ d_out, size_t num_segments) {
  __shared__ half d_out_s[16 * WMMA_TILE_SIZE];

  const int localWarpIdx        = threadIdx.x / WARP_SIZE;
  const int local_offset        = localWarpIdx * WMMA_TILE_SIZE;
  const int laneid              = threadIdx.x % WARP_SIZE;
  const size_t globalSegmentIdx = blockIdx.x * WARPS_PER_BLOCK + localWarpIdx;
  const size_t global_offset    = globalSegmentIdx * SEGMENT_SIZE;

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_b::set_first_row_ones(r_frag);

  wmma::fill_fragment(out_frag, zero<half>());

#pragma unroll min_unroll(SEGMENT_SIZE / 16, 32)
  for (size_t ii = 0; ii < SEGMENT_SIZE / 16; ii++) {
    const size_t offset = global_offset + ii * 16;

    // WARPS_PER_BLOCK * SEGMENT_SIZE cannot be more than 2^31 - 1 = 2147483647
    wmma::load_matrix_sync(a_frag, d_in + offset, WARPS_PER_BLOCK * SEGMENT_SIZE);

    wmma::mma_sync(out_frag, a_frag, r_frag, out_frag);
  }

  utils::fragment::matrix_c::store_first_row(&d_out_s[local_offset], out_frag);

  if (laneid < 16) {
    d_out[globalSegmentIdx + laneid * WARPS_PER_BLOCK] = d_out_s[local_offset + laneid];
  }
}

// each block calculates consecutive 16 segments
template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_reduction_16n_block(
    const half *__restrict__ d_in, half *__restrict__ d_out, size_t num_segments) {

  __shared__ half d_out_s[16 * WMMA_TILE_SIZE];

  const size_t global_offset =
      p2_mul<16L * SEGMENT_SIZE>(static_cast<size_t>(blockIdx.x));
  const size_t wmma_tiles_per_segment = p2_div<16>(SEGMENT_SIZE);
  const int localWarpIdx              = p2_div<WARP_SIZE>(threadIdx.x);
  const int local_offset              = p2_mul<WMMA_TILE_SIZE>(localWarpIdx);
  const int laneid                    = p2_mod<WARP_SIZE>(threadIdx.x);

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::col_major> r_t_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_b::set_first_row_ones(r_frag);
  utils::fragment::matrix_a::set_first_column_ones(
      r_t_frag); // not sure to set col or set row

  wmma::fill_fragment(out_frag, zero<half>());

#pragma unroll
  for (size_t ii = 0; ii < wmma_tiles_per_segment; ii += WARPS_PER_BLOCK) {
    const size_t offset = global_offset + (localWarpIdx + ii) * 16;

    wmma::load_matrix_sync(a_frag, d_in + offset, SEGMENT_SIZE);

    wmma::mma_sync(out_frag, a_frag, r_frag, out_frag);
  }

  utils::fragment::matrix_c::store_first_column(d_out_s + local_offset, out_frag);
  // wmma::store_matrix_sync(d_out_s + local_offset, out_frag, 16, wmma::mem_col_major);

  if (localWarpIdx == 0) {
    wmma::fill_fragment(out_frag, zero<half>());
    wmma::load_matrix_sync(b_frag, d_out_s, 256);
    wmma::mma_sync(out_frag, r_t_frag, b_frag, out_frag);

    utils::fragment::matrix_c::store_first_row(&d_out[blockIdx.x * 16], out_frag);
    // wmma::store_matrix_sync(d_out_s, out_frag, 16, wmma::mem_row_major);
    // if (laneid < 16) {
    //   d_out[blockIdx.x * 16 + laneid] = d_out_s[laneid];
    // }
  }
}

// segment_size = WMMA_TILE_SIZE
// each warp calculates SEGMENTS_PER_WARP segments
template <int SEGMENTS_PER_WARP, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_reduction_256(const half *__restrict__ d_in,
                                                            half *__restrict__ d_out,
                                                            size_t num_segments) {

  __shared__ half ra_mat_s[16 * WMMA_TILE_SIZE];

  const int localWarpIdx = p2_div<WARP_SIZE>(threadIdx.x);
  const int laneid       = p2_mod<WARP_SIZE>(threadIdx.x);

  const size_t globalWarpIdx = (blockIdx.x * BLOCK_DIM + threadIdx.x) / WARP_SIZE;
  const size_t local_offset  = p2_mul<WMMA_TILE_SIZE>(localWarpIdx);

  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> r_t_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> ra_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> ra_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_a::set_first_row_ones(r_frag);
  utils::fragment::matrix_b::set_first_column_ones(r_t_frag);

#pragma unroll
  for (int ii = 0; ii < SEGMENTS_PER_WARP; ii++) {
    const size_t globalSegmentIdx = globalWarpIdx * SEGMENTS_PER_WARP + ii;
    const size_t offset           = p2_mul<WMMA_TILE_SIZE>(globalSegmentIdx);

    wmma::fill_fragment(ra_frag, zero<half>());
    wmma::fill_fragment(out_frag, zero<half>());
    wmma::load_matrix_sync(a_frag, d_in + offset, 16);

    wmma::mma_sync(ra_frag, r_frag, a_frag, ra_frag);

    // store accumulator ra_frag into shared memory and load it into
    // matrix_a fragment ra_mat_frag
    wmma::store_matrix_sync(ra_mat_s + local_offset, ra_frag, 16, wmma::mem_row_major);
    wmma::load_matrix_sync(ra_mat_frag, ra_mat_s + local_offset, 16);

    wmma::mma_sync(out_frag, ra_mat_frag, r_t_frag, out_frag);

    utils::fragment::matrix_c::store_first_element(&d_out[globalSegmentIdx], out_frag);
  }
}

// each warp calculates 1 segment
template <typename OUT_TYPE, size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_reduction_256n(
    const half *__restrict__ d_in, OUT_TYPE *__restrict__ d_out, size_t num_segments) {

  __shared__ half ra_mat_s[16 * WMMA_TILE_SIZE];

  const int localWarpIdx = p2_div<WARP_SIZE>(threadIdx.x);
  const int laneid       = p2_mod<WARP_SIZE>(threadIdx.x);

  const size_t globalWarpIdx  = (blockIdx.x * BLOCK_DIM + threadIdx.x) / WARP_SIZE;
  const size_t global_offset  = globalWarpIdx * SEGMENT_SIZE;
  const size_t local_offset   = p2_mul<WMMA_TILE_SIZE>(localWarpIdx);
  const size_t num_wmma_tiles = (SEGMENT_SIZE + WMMA_TILE_SIZE - 1) / WMMA_TILE_SIZE;

  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> r_t_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> ra_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> ra_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, OUT_TYPE> out_frag;

  utils::fragment::matrix_a::set_first_row_ones(r_frag);
  utils::fragment::matrix_b::set_first_column_ones(r_t_frag);

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

  utils::fragment::matrix_c::store_first_element(&d_out[globalWarpIdx], out_frag);
}

// each block calculates 1 segment
template <typename OUT_TYPE, size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_reduction_256n_block(
    const half *__restrict__ d_in, OUT_TYPE *__restrict__ d_out, size_t num_segments) {

  __shared__ OUT_TYPE ra_mat_s[16 * WMMA_TILE_SIZE];
  __shared__ OUT_TYPE d_out_s[WMMA_TILE_SIZE];

  const size_t globalSegmentIdx       = blockIdx.x;
  const size_t wmma_tiles_per_segment = SEGMENT_SIZE / WMMA_TILE_SIZE;
  const size_t global_offset          = globalSegmentIdx * SEGMENT_SIZE;
  const int localWarpIdx              = threadIdx.x / WARP_SIZE;
  const int local_offset              = localWarpIdx * WMMA_TILE_SIZE;
  const int laneid                    = threadIdx.x % WARP_SIZE;

  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> r_t_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> ra_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> ra_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, OUT_TYPE> out_frag;

  utils::fragment::matrix_a::set_first_row_ones(r_frag);
  utils::fragment::matrix_b::set_first_column_ones(r_t_frag);

  // TODO: r_mat_s should be initialized to 0

  wmma::fill_fragment(out_frag, zero<OUT_TYPE>());
  wmma::fill_fragment(ra_frag, zero<half>());

#pragma unroll
  for (size_t ii = 0; ii < wmma_tiles_per_segment; ii += WARPS_PER_BLOCK) {
    const size_t offset = global_offset + (ii + localWarpIdx) * WMMA_TILE_SIZE;
    wmma::load_matrix_sync(a_frag, d_in + offset, 16);
    wmma::mma_sync(ra_frag, r_frag, a_frag, ra_frag);
  }

  utils::fragment::matrix_c::store_first_row(ra_mat_s + local_offset, ra_frag);
  // wmma::store_matrix_sync(ra_mat_s + local_offset, ra_frag, 16, wmma::mem_row_major);

  if (localWarpIdx == 0) {
    wmma::load_matrix_sync(ra_mat_frag, ra_mat_s, 256);
    wmma::mma_sync(out_frag, ra_mat_frag, r_t_frag, out_frag);

    utils::fragment::matrix_c::store_first_column(d_out_s, out_frag);
    // wmma::store_matrix_sync(d_out_s, out_frag, 16, wmma::mem_row_major);

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

  __shared__ OUT_TYPE ra_mat_s[16 * WMMA_TILE_SIZE];
  __shared__ OUT_TYPE d_out_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];

  const size_t globalSegmentIdx       = blockIdx.x;
  const size_t wmma_tiles_per_segment = SEGMENT_SIZE / WMMA_TILE_SIZE;
  const size_t global_offset          = globalSegmentIdx * SEGMENT_SIZE;
  const int localWarpIdx              = threadIdx.x / WARP_SIZE;
  const int local_offset              = localWarpIdx * WMMA_TILE_SIZE;
  const int laneid                    = threadIdx.x % WARP_SIZE;

  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> r_t_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> ra_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> ra_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, OUT_TYPE> out_frag;

  utils::fragment::matrix_a::set_first_row_ones(r_frag);
  utils::fragment::matrix_b::set_first_column_ones(r_t_frag);

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

  utils::fragment::matrix_c::store_first_element(&d_out_s[localWarpIdx], out_frag);

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
  __shared__ half ra_mat_s[16 * WMMA_TILE_SIZE];
  __shared__ half d_out_s[WARPS_PER_BLOCK];

  const int localWarpIdx = p2_div<WARP_SIZE>(threadIdx.x);
  const int laneid       = p2_mod<WARP_SIZE>(threadIdx.x);

  const size_t globalThreadIdx = blockIdx.x * BLOCK_DIM + threadIdx.x;
  const size_t globalWarpIdx   = globalThreadIdx / WARP_SIZE;
  const size_t global_offset   = globalWarpIdx * SEGMENT_SIZE;
  const size_t local_offset    = p2_mul<WMMA_TILE_SIZE>(localWarpIdx);
  const size_t num_wmma_tiles  = (SEGMENT_SIZE + WMMA_TILE_SIZE - 1) / WMMA_TILE_SIZE;

  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> r_t_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> ra_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> ra_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_a::set_first_row_ones(r_frag);
  utils::fragment::matrix_b::set_first_column_ones(r_t_frag);

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

  utils::fragment::matrix_c::store_first_element(&d_out_s[local_offset], out_frag);

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

  __shared__ half ra_mat_s[16 * WMMA_TILE_SIZE];
  __shared__ half d_out_s[WARPS_PER_BLOCK];

  const int localWarpIdx = p2_div<WARP_SIZE>(threadIdx.x);
  const int laneid       = p2_mod<WARP_SIZE>(threadIdx.x);

  const size_t globalThreadIdx = blockIdx.x * BLOCK_DIM + threadIdx.x;
  const size_t globalWarpIdx   = globalThreadIdx / WARP_SIZE;
  const size_t global_offset   = globalWarpIdx * SEGMENT_SIZE;
  const int local_offset       = p2_mul<WMMA_TILE_SIZE>(localWarpIdx);
  const size_t num_wmma_tiles  = (SEGMENT_SIZE + WMMA_TILE_SIZE - 1) / WMMA_TILE_SIZE;

  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> r_t_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> ra_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> ra_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_a::set_first_row_ones(r_frag);
  utils::fragment::matrix_b::set_first_column_ones(r_t_frag);

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

  utils::fragment::matrix_c::store_first_element(&d_out_s[local_offset], out_frag);

  // local reduction
  if (laneid == 0 && threadIdx.x != 0) {
    xprintf("d_out_s[%d] = %f -- d_out[0] = %f\n", local_offset,
            float(d_out_s[local_offset]), float(d_out[0]));
    atomicAdd(&d_out_s[0], d_out_s[local_offset]);
  }
  __syncthreads();
  // global reduction
  if (threadIdx.x == 0) {
    atomicAdd(&d_out[0], d_out_s[0]);
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
  __shared__ half dummy[6]; // to maintain alignment
  __shared__ half ra_mat_s[16 * WMMA_TILE_SIZE];
  __shared__ half d_out_s[WARPS_PER_BLOCK];

  (void) dummy;

  const int localWarpIdx = p2_div<WARP_SIZE>(threadIdx.x);
  const int laneid       = p2_mod<WARP_SIZE>(threadIdx.x);

  const size_t globalThreadIdx = blockIdx.x * BLOCK_DIM + threadIdx.x;
  const size_t globalWarpIdx   = globalThreadIdx / WARP_SIZE;
  const size_t global_offset   = globalWarpIdx * SEGMENT_SIZE;
  const int local_offset       = p2_mul<WMMA_TILE_SIZE>(localWarpIdx);
  const int num_warps          = BLOCK_DIM / WARP_SIZE;
  const size_t num_wmma_tiles  = (SEGMENT_SIZE + WMMA_TILE_SIZE - 1) / WMMA_TILE_SIZE;

  if (threadIdx.x == 0) {
    block_reduction_visitor = 0;
  }

  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> r_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::col_major> r_t_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> ra_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> ra_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_b::set_first_row_ones(r_frag);
  utils::fragment::matrix_b::set_first_column_ones(r_t_frag);

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

  utils::fragment::matrix_c::store_first_element(&d_out_s[local_offset], out_frag);

  if (laneid == 0) {
    if (threadIdx.x != 0) {
      atomicAdd(&d_out_s[0], d_out_s[local_offset]);
    }
    if (atomicInc(&block_reduction_visitor, num_warps) == (num_warps - 1)) {
      atomicAdd(&d_out[0], d_out_s[0]);
    }
  }
}
} // namespace wmma_unsafe_reduction
