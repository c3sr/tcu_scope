#pragma once

#include "init/init.hpp"
#include "utils/utils.hpp"

#include <cooperative_groups.h>
#include <mma.h>

namespace wmma_unsafe_prefixsum {

using namespace nvcuda;

using namespace cooperative_groups;

#ifndef WARP_SIZE
#define WARP_SIZE (32)
#endif

// MMA matrix tile dimensions. (16, 16, 16), (32, 8, 16), and (8, 32,
// 16) are currently supported.
static const int M              = 16;
static const int N              = 16;
static const int K              = 16;
static const int WMMA_TILE_SIZE = (M * N);

constexpr __host__ __device__ int min_unroll(size_t x, size_t y) {
  return x <= y ? x : y;
}

template <int BLOCK_DIM, size_t SEGMENT_SIZE>
static __global__ void add_partial_sums(half *__restrict__ output,
                                        const half *__restrict__ partial_sums,
                                        size_t num_elements) {
  const size_t global_offset = blockIdx.x * SEGMENT_SIZE;

#pragma unroll
  for (size_t ii = 0; ii < SEGMENT_SIZE; ii += BLOCK_DIM) {
    const size_t offset = global_offset + ii + threadIdx.x;
    output[offset] += partial_sums[blockIdx.x];
  }
}

// segment_size = 16
// each warp calculates WMMA_TILES_PER_WARP * 16 segments
template <int WMMA_TILES_PER_WARP, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_prefixsum_16(const half *__restrict__ d_in,
                                                           half *__restrict__ d_out,
                                                           size_t num_segments) {

  const size_t globalWarpIdx = p2_div<WARP_SIZE>(blockIdx.x * BLOCK_DIM + threadIdx.x);

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> u_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_b::set_upper_triangular(u_frag);

#pragma unroll
  for (int ii = 0; ii < WMMA_TILES_PER_WARP; ii++) {
    const int tileIdx          = globalWarpIdx * WMMA_TILES_PER_WARP + ii;
    const size_t global_offset = tileIdx * WMMA_TILE_SIZE;

    wmma::fill_fragment(out_frag, zero<half>());
    wmma::load_matrix_sync(a_frag, d_in + global_offset, 16);

    wmma::mma_sync(out_frag, a_frag, u_frag, out_frag);

    wmma::store_matrix_sync(d_out + global_offset, out_frag, 16, wmma::mem_row_major);
  }
}

// each warp calculates 16 consecutive segments
template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_prefixsum_16n(const half *__restrict__ d_in,
                                                            half *__restrict__ d_out,
                                                            size_t num_segments) {

  __shared__ half curr_out_frag_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];
  __shared__ half next_out_frag_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];

  const int localWarpIdx      = p2_div<WARP_SIZE>(threadIdx.x);
  const int laneid            = p2_mod<WARP_SIZE>(threadIdx.x);
  const size_t globalWarpIdx  = p2_div<WARP_SIZE>(blockIdx.x * BLOCK_DIM + threadIdx.x);
  const size_t segment_offset = globalWarpIdx * 16;
  const size_t global_offset  = segment_offset * SEGMENT_SIZE;
  const int local_offset      = localWarpIdx * WMMA_TILE_SIZE;

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> u_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_b::set_upper_triangular(u_frag);
  wmma::fill_fragment(out_frag, zero<half>());

#pragma unroll min_unroll(SEGMENT_SIZE / 16, 32)
  for (int ii = 0; ii < p2_div<16>(SEGMENT_SIZE); ii++) {
    const bool is_not_last_iteration = (ii + 1) < p2_div<16>(SEGMENT_SIZE);
    const int offset                 = global_offset + ii * 16;

    wmma::load_matrix_sync(a_frag, d_in + offset, SEGMENT_SIZE);

    wmma::mma_sync(out_frag, a_frag, u_frag, out_frag);

    wmma::store_matrix_sync(curr_out_frag_s + local_offset, out_frag, 16,
                            wmma::mem_row_major);

    __syncthreads();

#pragma unroll
    for (int jj = 0; jj < WMMA_TILE_SIZE; jj += WARP_SIZE) {
      const auto idx = jj + laneid;
      const auto m   = p2_div<16>(idx);
      const auto n   = p2_mod<16>(idx);
      d_out[global_offset + m * SEGMENT_SIZE + ii * 16 + n] =
          curr_out_frag_s[local_offset + idx];
      if (is_not_last_iteration) {
        next_out_frag_s[local_offset + idx] = curr_out_frag_s[local_offset + 16 * m + 15];
      }
    }

    if (is_not_last_iteration) {
      __syncthreads();

      wmma::load_matrix_sync(out_frag, next_out_frag_s + local_offset, 16,
                             wmma::mem_row_major);
    }
  }
}

// each block calculates consecutive 16 segments
template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_prefixsum_16n_block(
    const half *__restrict__ d_in, half *__restrict__ d_out, size_t num_segments) {

  __shared__ half curr_out_frag_s[16 * WMMA_TILE_SIZE];
  __shared__ half next_out_frag_s[16 * WMMA_TILE_SIZE];
  __shared__ half partial_sums_s[WMMA_TILE_SIZE];

  const int localWarpIdx              = p2_div<WARP_SIZE>(threadIdx.x);
  const int laneid                    = p2_mod<WARP_SIZE>(threadIdx.x);
  const size_t wmma_tiles_per_segment = p2_div<16>(SEGMENT_SIZE);
  const size_t global_offset          = blockIdx.x * 16 * SEGMENT_SIZE;
  const int local_offset              = localWarpIdx * WMMA_TILE_SIZE;
  const size_t globalWarpIdx = p2_div<WARP_SIZE>(blockIdx.x * BLOCK_DIM + threadIdx.x);

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> u_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_b::set_upper_triangular(u_frag);

#pragma unroll
  for (size_t ii = 0; ii < wmma_tiles_per_segment; ii += WARPS_PER_BLOCK) {
    const bool is_not_last_iteration = (ii + WARPS_PER_BLOCK) < wmma_tiles_per_segment;
    const size_t offset              = global_offset + (localWarpIdx + ii) * 16;

    wmma::fill_fragment(out_frag, zero<half>());

    wmma::load_matrix_sync(a_frag, d_in + offset, SEGMENT_SIZE);

    wmma::mma_sync(out_frag, a_frag, u_frag, out_frag);

    wmma::store_matrix_sync(curr_out_frag_s + local_offset, out_frag, 16,
                            wmma::mem_row_major);

    if (localWarpIdx == 0) {
      wmma::fill_fragment(out_frag, zero<half>());
      wmma::load_matrix_sync(
          b_frag, curr_out_frag_s + 16,
          16); // TODO: should be curr_out_frag_s + 15(fix when 128 alignment is relaxed)
      wmma::mma_sync(out_frag, b_frag, u_frag, out_frag);
      wmma::store_matrix_sync(partial_sums_s, out_frag, 16, wmma::mem_row_major);
    }

    __syncthreads();

#pragma unroll
    for (int jj = 0; jj < WMMA_TILE_SIZE; jj += WARP_SIZE) {
      const auto idx = jj + laneid;
      const auto m   = p2_div<16>(idx);
      const auto n   = p2_mod<16>(idx);
      half val       = curr_out_frag_s[local_offset + idx];
      if (localWarpIdx > 0) {
        val += partial_sums_s[m * 16 + localWarpIdx - 1];
      }
      d_out[global_offset + m * SEGMENT_SIZE + ii * 16 + n] = val;
      if (is_not_last_iteration) {
        next_out_frag_s[local_offset + idx] = partial_sums_s[16 * m + 15];
      }
    }

    if (is_not_last_iteration) {
      __syncthreads();

      wmma::load_matrix_sync(out_frag, next_out_frag_s + local_offset, 16,
                             wmma::mem_row_major);
    }
  }
}

// each warp calculates 16 strided segments
template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_prefixsum_16n_opt(
    const half *__restrict__ d_in, half *__restrict__ d_out, size_t num_segments) {

  __shared__ half curr_out_frag_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];
  __shared__ half next_out_frag_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];

  const int localWarpIdx        = p2_div<WARP_SIZE>(threadIdx.x);
  const int laneid              = p2_mod<WARP_SIZE>(threadIdx.x);
  const int local_offset        = localWarpIdx * WMMA_TILE_SIZE;
  const size_t globalSegmentIdx = blockIdx.x * WARPS_PER_BLOCK + localWarpIdx;
  const size_t global_offset    = globalSegmentIdx * SEGMENT_SIZE;

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> u_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_b::set_upper_triangular(u_frag);
  wmma::fill_fragment(out_frag, zero<half>());

#pragma unroll min_unroll(SEGMENT_SIZE / 16, 32)
  for (int ii = 0; ii < p2_div<16>(SEGMENT_SIZE); ii++) {
    const bool is_not_last_iteration = (ii + 1) < p2_div<16>(SEGMENT_SIZE);
    const int offset                 = global_offset + ii * 16;

    // WARPS_PER_BLOCK * SEGMENT_SIZE cannot be more than 2^31 - 1 = 2147483647
    wmma::load_matrix_sync(a_frag, d_in + offset, WARPS_PER_BLOCK * SEGMENT_SIZE);

    wmma::mma_sync(out_frag, a_frag, u_frag, out_frag);

    wmma::store_matrix_sync(curr_out_frag_s + local_offset, out_frag, 16,
                            wmma::mem_row_major);

#pragma unroll
    for (int jj = 0; jj < WMMA_TILE_SIZE; jj += WARP_SIZE) {
      const auto idx = jj + laneid;
      const auto m   = p2_div<16>(idx);
      const auto n   = p2_mod<16>(idx);
      d_out[global_offset + m * WARPS_PER_BLOCK * SEGMENT_SIZE + ii * 16 + n] =
          curr_out_frag_s[local_offset + idx];
      if (is_not_last_iteration) {
        next_out_frag_s[local_offset + idx] = curr_out_frag_s[local_offset + 16 * m + 15];
      }
    }

    if (is_not_last_iteration) {
      __syncthreads();

      wmma::load_matrix_sync(out_frag, next_out_frag_s + local_offset, 16,
                             wmma::mem_row_major);
    }
  }
}

// segment_size = 256
// each warp calculates SEGMENTS_PER_WARP segments
template <int SEGMENTS_PER_WARP, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_prefixsum_256(const half *__restrict__ d_in,
                                                            half *__restrict__ d_out,
                                                            size_t num_segments) {
  __shared__ half la_mat_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];

  const int localWarpIdx     = p2_div<WARP_SIZE>(threadIdx.x);
  const int laneid           = p2_mod<WARP_SIZE>(threadIdx.x);
  const size_t globalWarpIdx = p2_div<WARP_SIZE>(blockIdx.x * BLOCK_DIM + threadIdx.x);
  const int local_offset     = localWarpIdx * WMMA_TILE_SIZE;

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> u_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> l_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> o_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> la_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> la_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> au_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_b::set_upper_triangular(u_frag);
  utils::fragment::matrix_a::set_strict_lower_triangular(l_frag);

  wmma::fill_fragment(o_frag, one<half>());

#pragma unroll
  for (int ii = 0; ii < SEGMENTS_PER_WARP; ii++) {
    const size_t segment_idx   = globalWarpIdx * SEGMENTS_PER_WARP + ii;
    const size_t global_offset = segment_idx * WMMA_TILE_SIZE;

    wmma::fill_fragment(au_frag, zero<half>());
    wmma::fill_fragment(la_frag, zero<half>());
    // wmma::fill_fragment(out_frag, zero<half>()); // not needed (OPTIMIZATON)
    wmma::load_matrix_sync(a_frag, d_in + global_offset, 16);
    wmma::load_matrix_sync(b_frag, d_in + global_offset, 16);

    wmma::mma_sync(au_frag, a_frag, u_frag, au_frag);

    wmma::mma_sync(la_frag, l_frag, b_frag, la_frag);

    // store accumulator la_frag into shared memory and load it into
    // matrix_a fragment la_mat_frag
    wmma::store_matrix_sync(la_mat_s + local_offset, la_frag, 16, wmma::mem_row_major);
    wmma::load_matrix_sync(la_mat_frag, la_mat_s + local_offset, 16);

    wmma::mma_sync(out_frag, la_mat_frag, o_frag, au_frag);

    wmma::store_matrix_sync(d_out + global_offset, out_frag, 16, wmma::mem_row_major);
  }
}

// segment_size = 256
// each warp calculates SEGMENTS_PER_WARP segments and writes
// SEGMENTS_PER_WARP
// partial sums
template <int SEGMENTS_PER_WARP, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_prefixsum_256_ps(
    const half *__restrict__ d_in, half *__restrict__ d_out,
    half *__restrict__ partial_sums, size_t num_segments) {
  __shared__ half la_mat_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];

  const int localWarpIdx     = p2_div<WARP_SIZE>(threadIdx.x);
  const int laneid           = p2_mod<WARP_SIZE>(threadIdx.x);
  const size_t globalWarpIdx = p2_div<WARP_SIZE>(blockIdx.x * BLOCK_DIM + threadIdx.x);
  const int local_offset     = localWarpIdx * WMMA_TILE_SIZE;

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> u_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> l_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> o_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> la_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> la_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> au_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_b::set_upper_triangular(u_frag);
  utils::fragment::matrix_a::set_strict_lower_triangular(l_frag);

  wmma::fill_fragment(o_frag, one<half>());

#pragma unroll
  for (int ii = 0; ii < SEGMENTS_PER_WARP; ii++) {
    const size_t globalSegmentIdx = globalWarpIdx * SEGMENTS_PER_WARP + ii;
    const size_t offset           = globalSegmentIdx * WMMA_TILE_SIZE;

    wmma::fill_fragment(au_frag, zero<half>());
    wmma::fill_fragment(la_frag, zero<half>());
    wmma::fill_fragment(out_frag, zero<half>());

    wmma::load_matrix_sync(a_frag, d_in + offset, 16);
    wmma::load_matrix_sync(b_frag, d_in + offset, 16);

    wmma::mma_sync(au_frag, a_frag, u_frag, au_frag);

    wmma::mma_sync(la_frag, l_frag, b_frag, la_frag);

    // store accumulator la_frag into shared memory and load it into
    // matrix_a fragment la_mat_frag
    wmma::store_matrix_sync(la_mat_s + local_offset, la_frag, 16, wmma::mem_row_major);
    wmma::load_matrix_sync(la_mat_frag, la_mat_s + local_offset, 16);

    wmma::mma_sync(out_frag, la_mat_frag, o_frag, au_frag);

    const half partial_sum =
        utils::fragment::matrix_c::last_element<WARPS_PER_BLOCK>(out_frag);

    // Store the output
    wmma::store_matrix_sync(d_out + offset, out_frag, 16, wmma::mem_row_major);

    if (laneid == 0) {
      partial_sums[globalSegmentIdx] = partial_sum;
    }
  }
}

// SEGMENTS_PER_WARP = 1
// each warp calculates 1 segment
template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void
    compute_wmma_segmented_prefixsum_256n(const half *__restrict__ d_in,
                                          half *__restrict__ d_out, size_t num_segments) {

  __shared__ half la_mat_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];

  const int localWarpIdx      = p2_div<WARP_SIZE>(threadIdx.x);
  const int laneid            = p2_mod<WARP_SIZE>(threadIdx.x);
  const size_t globalWarpIdx  = p2_div<WARP_SIZE>(blockIdx.x * BLOCK_DIM + threadIdx.x);
  const int local_offset      = localWarpIdx * WMMA_TILE_SIZE;
  const size_t global_offset  = globalWarpIdx * SEGMENT_SIZE;
  const size_t num_wmma_tiles = p2_div<WMMA_TILE_SIZE>(SEGMENT_SIZE + WMMA_TILE_SIZE - 1);

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> u_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> l_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> o_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> la_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> la_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> au_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_b::set_upper_triangular(u_frag);
  utils::fragment::matrix_a::set_strict_lower_triangular(l_frag);
  wmma::fill_fragment(o_frag, one<half>());
  wmma::fill_fragment(out_frag, zero<half>());

  half partial_sum = zero<half>();

#pragma unroll
  for (size_t ii = 0; ii < num_wmma_tiles; ii++) {
    const size_t offset = global_offset + ii * WMMA_TILE_SIZE;

    wmma::fill_fragment(la_frag, zero<half>());
    wmma::load_matrix_sync(a_frag, d_in + offset, 16);
    wmma::load_matrix_sync(b_frag, d_in + offset, 16);

    wmma::mma_sync(au_frag, a_frag, u_frag, out_frag);
    wmma::mma_sync(la_frag, l_frag, b_frag, la_frag);

    // store accumulator la_frag into shared memory and load it into
    // matrix_a fragment la_mat_frag
    wmma::store_matrix_sync(la_mat_s + local_offset, la_frag, 16, wmma::mem_row_major);
    wmma::load_matrix_sync(la_mat_frag, la_mat_s + local_offset, 16);

    wmma::mma_sync(out_frag, la_mat_frag, o_frag, au_frag);

    partial_sum = utils::fragment::matrix_c::last_element<WARPS_PER_BLOCK>(out_frag);
    wmma::fill_fragment(out_frag, partial_sum);

    wmma::store_matrix_sync(d_out + offset, out_frag, 16, wmma::mem_row_major);
  }
}

// SEGMENTS_PER_WARP = 1
// each warp calculates 1 segment and writes 1 partial sum
template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void
    compute_wmma_segmented_prefixsum_256n_ps(const half *__restrict__ d_in,
                                             half *__restrict__ d_out,
                                             half *__restrict__ partial_sums,
                                             size_t num_segments) {

  __shared__ half u_frag_s[WMMA_TILE_SIZE];
  __shared__ half l_frag_s[WMMA_TILE_SIZE];
  __shared__ half la_mat_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];

  const int localWarpIdx      = p2_div<WARP_SIZE>(threadIdx.x);
  const int laneid            = p2_mod<WARP_SIZE>(threadIdx.x);
  const size_t globalWarpIdx  = p2_div<WARP_SIZE>(blockIdx.x * BLOCK_DIM + threadIdx.x);
  const int local_offset      = localWarpIdx * WMMA_TILE_SIZE;
  const size_t global_offset  = globalWarpIdx * SEGMENT_SIZE;
  const size_t num_wmma_tiles = p2_div<WMMA_TILE_SIZE>(SEGMENT_SIZE + WMMA_TILE_SIZE - 1);

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> u_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> l_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> o_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> la_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> la_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> au_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_b::set_upper_triangular(u_frag);
  utils::fragment::matrix_a::set_strict_lower_triangular(l_frag);
  wmma::fill_fragment(o_frag, one<half>());
  wmma::fill_fragment(out_frag, zero<half>());

  half partial_sum = zero<half>();

#pragma unroll
  for (size_t ii = 0; ii < num_wmma_tiles; ii++) {
    const size_t offset = global_offset + ii * WMMA_TILE_SIZE;

    wmma::fill_fragment(la_frag, zero<half>());
    wmma::load_matrix_sync(a_frag, d_in + offset, 16);
    wmma::load_matrix_sync(b_frag, d_in + offset, 16);

    wmma::mma_sync(au_frag, a_frag, u_frag, out_frag);
    wmma::mma_sync(la_frag, l_frag, b_frag, la_frag);

    // store accumulator la_frag into shared memory and load it into
    // matrix_a fragment la_mat_frag
    wmma::store_matrix_sync(la_mat_s + local_offset, la_frag, 16, wmma::mem_row_major);
    wmma::load_matrix_sync(la_mat_frag, la_mat_s + local_offset, 16);

    wmma::mma_sync(out_frag, la_mat_frag, o_frag, au_frag);

    partial_sum = utils::fragment::matrix_c::last_element<WARPS_PER_BLOCK>(out_frag);
    wmma::fill_fragment(out_frag, partial_sum);

    wmma::store_matrix_sync(d_out + offset, out_frag, 16, wmma::mem_row_major);
  }

  if (laneid == 0) {
    partial_sums[globalWarpIdx] = partial_sum;
  }
}

// SEGMENTS_PER_WARP = 1
// each warp calculates 1 segment
template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_prefixsum_256n_block(
    const half *__restrict__ d_in, half *__restrict__ d_out, size_t num_segments) {

  __shared__ half la_mat_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];
  __shared__ half out_frag_s[16 * WMMA_TILE_SIZE];
  __shared__ half partial_sums_s[WMMA_TILE_SIZE]; // only use the first 16

  const size_t globalSegmentIdx = blockIdx.x;
  const int localWarpIdx        = p2_div<WARP_SIZE>(threadIdx.x);
  const int laneid              = p2_mod<WARP_SIZE>(threadIdx.x);
  const size_t globalWarpIdx    = p2_div<WARP_SIZE>(blockIdx.x * BLOCK_DIM + threadIdx.x);
  const int local_offset        = localWarpIdx * WMMA_TILE_SIZE;
  const size_t global_offset    = globalSegmentIdx * SEGMENT_SIZE;
  const size_t wmma_tiles_per_segment = p2_div<WMMA_TILE_SIZE>(SEGMENT_SIZE);

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::col_major> c_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> u_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> l_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> o_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> la_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> la_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> au_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_b::set_upper_triangular(u_frag);
  utils::fragment::matrix_a::set_strict_lower_triangular(l_frag);
  wmma::fill_fragment(o_frag, one<half>());
  wmma::fill_fragment(out_frag, zero<half>());

#pragma unroll
  for (size_t ii = 0; ii < wmma_tiles_per_segment; ii += WARPS_PER_BLOCK) {
    const size_t offset = global_offset + (ii + localWarpIdx) * WMMA_TILE_SIZE;

    wmma::fill_fragment(la_frag, zero<half>());
    wmma::load_matrix_sync(a_frag, d_in + offset, 16);
    wmma::load_matrix_sync(b_frag, d_in + offset, 16);

    wmma::mma_sync(au_frag, a_frag, u_frag, out_frag);
    wmma::mma_sync(la_frag, l_frag, b_frag, la_frag);

    // store accumulator la_frag into shared memory and load it into
    // matrix_a
    // fragment la_mat_frag
    wmma::store_matrix_sync(la_mat_s + local_offset, la_frag, 16, wmma::mem_row_major);
    wmma::load_matrix_sync(la_mat_frag, la_mat_s + local_offset, 16);

    wmma::mma_sync(out_frag, la_mat_frag, o_frag, au_frag);

    wmma::store_matrix_sync(out_frag_s + local_offset, out_frag, 16, wmma::mem_row_major);

    if (localWarpIdx == 0) {
      wmma::fill_fragment(out_frag, zero<half>());
      wmma::load_matrix_sync(c_frag, out_frag_s + 256,
                             WMMA_TILE_SIZE); // TODO: should be out_frag_s + 255(fix
                                              // when 128 alignment is relaxed)
      wmma::mma_sync(out_frag, c_frag, u_frag, out_frag);
      wmma::store_matrix_sync(partial_sums_s, out_frag, 16, wmma::mem_row_major);
    }

    __syncthreads();

#pragma unroll
    for (int jj = 0; jj < WMMA_TILE_SIZE; jj += WARP_SIZE) {
      const auto idx = jj + laneid;
      const auto m   = p2_div<16>(idx);
      const auto n   = p2_mod<16>(idx);
      half val       = out_frag_s[local_offset + idx];
      if (localWarpIdx > 0) {
        val += partial_sums_s[localWarpIdx - 1];
      }
      d_out[offset + idx] = val;
    }

    wmma::fill_fragment(out_frag, partial_sums_s[15]);
  }
}

// SEGMENTS_PER_WARP = 1
// each warp calculates 1 segment
template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK, int BLOCK_DIM>
static __global__ void compute_wmma_segmented_prefixsum_256n_block_ps(
    const half *__restrict__ d_in, half *__restrict__ d_out, size_t num_segments) {

  __shared__ half la_mat_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];
  __shared__ half out_frag_s[16 * WMMA_TILE_SIZE];
  __shared__ half partial_sums[WMMA_TILE_SIZE]; // only use the first 16

  const int localWarpIdx              = p2_div<WARP_SIZE>(threadIdx.x);
  const int local_offset              = localWarpIdx * WMMA_TILE_SIZE;
  const int laneid                    = p2_mod<WARP_SIZE>(threadIdx.x);
  const size_t globalSegmentIdx       = blockIdx.x;
  const size_t wmma_tiles_per_segment = p2_div<WMMA_TILE_SIZE>(SEGMENT_SIZE);
  const size_t global_offset          = globalSegmentIdx * SEGMENT_SIZE;

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::col_major> c_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> u_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> l_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> o_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> la_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> la_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> au_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_b::set_upper_triangular(u_frag);
  utils::fragment::matrix_a::set_strict_lower_triangular(l_frag);
  wmma::fill_fragment(o_frag, one<half>());
  wmma::fill_fragment(out_frag, zero<half>());

  half partial_sum = zero<half>();

#pragma unroll
  for (size_t ii = 0; ii < wmma_tiles_per_segment; ii += WARPS_PER_BLOCK) {
    const bool is_not_last_iteration = (ii + WARPS_PER_BLOCK) < wmma_tiles_per_segment;
    const size_t offset = global_offset + (ii + localWarpIdx) * WMMA_TILE_SIZE;

    wmma::fill_fragment(la_frag, zero<half>());
    wmma::load_matrix_sync(a_frag, d_in + offset, 16);
    wmma::load_matrix_sync(b_frag, d_in + offset, 16);

    wmma::mma_sync(au_frag, a_frag, u_frag, out_frag);
    wmma::mma_sync(la_frag, l_frag, b_frag, la_frag);

    // store accumulator la_frag into shared memory and load it into
    // matrix_a
    // fragment la_mat_frag
    wmma::store_matrix_sync(la_mat_s + local_offset, la_frag, 16, wmma::mem_row_major);
    wmma::load_matrix_sync(la_mat_frag, la_mat_s + local_offset, 16);

    wmma::mma_sync(out_frag, la_mat_frag, o_frag, au_frag);

    wmma::store_matrix_sync(out_frag_s + local_offset, out_frag, 16, wmma::mem_row_major);

    if (localWarpIdx == 0) {
      wmma::fill_fragment(out_frag, zero<half>());
      wmma::load_matrix_sync(c_frag, out_frag_s + 256,
                             WMMA_TILE_SIZE); // TODO: should be curr_out_frag_s + 255(fix
                                              // when 128 alignment is relaxed)
      wmma::mma_sync(out_frag, c_frag, u_frag, out_frag);
      wmma::store_matrix_sync(partial_sums, out_frag, 16, wmma::mem_row_major);
    }

    __syncthreads();

#pragma unroll
    for (int jj = 0; jj < WMMA_TILE_SIZE; jj += WARP_SIZE) {
      const auto idx = jj + laneid;
      const auto m   = p2_div<16>(idx);
      const auto n   = p2_mod<16>(idx);
      half val       = out_frag_s[local_offset + idx];
      if (localWarpIdx > 0) {
        val += partial_sums[localWarpIdx - 1];
      }
      partial_sum         = val;
      d_out[offset + idx] = val;
    }
    if (is_not_last_iteration) {
      wmma::fill_fragment(out_frag, partial_sums[15]);
    }
  }

  if (threadIdx.x == BLOCK_DIM - 1) {
    partial_sums[globalSegmentIdx] = partial_sum;
  }
}

// SEGMENTS_PER_WARP = 1
// each warp calculates 1 segment
template <int WARPS_PER_BLOCK, int BLOCK_DIM, size_t SEGMENT_SIZE>
static __global__ void compute_wmma_prefixsum_cg(const half *__restrict__ d_in,
                                                 half *__restrict__ d_out,
                                                 half *__restrict__ partial_sums,
                                                 size_t num_segments) {
  __shared__ half la_mat_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];
  __shared__ half d_out_s[WARPS_PER_BLOCK * SEGMENT_SIZE];
  __shared__ half adjusted_sums[32]; // WARPS_PER_BLOCK

  const int localWarpIdx = p2_div<WARP_SIZE>(threadIdx.x);
  const int laneid       = p2_mod<WARP_SIZE>(threadIdx.x);

  const size_t globalWarpIdx = (blockIdx.x * BLOCK_DIM + threadIdx.x) / WARP_SIZE;
  const size_t global_offset = globalWarpIdx * SEGMENT_SIZE;
  const int local_offset     = localWarpIdx * WMMA_TILE_SIZE;
  const int d_out_s_offset   = localWarpIdx * SEGMENT_SIZE;

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> u_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> l_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> o_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> la_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> la_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> au_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_b::set_upper_triangular(u_frag);
  utils::fragment::matrix_a::set_strict_lower_triangular(l_frag);

  wmma::fill_fragment(o_frag, one<half>());
  wmma::fill_fragment(out_frag, zero<half>());

  half partial_sum = zero<half>();

#pragma unroll
  for (size_t ii = 0; ii < SEGMENT_SIZE / WMMA_TILE_SIZE; ii++) {
    const size_t in_offset  = global_offset + ii * WMMA_TILE_SIZE;
    const size_t out_offset = d_out_s_offset + ii * WMMA_TILE_SIZE;

    wmma::fill_fragment(la_frag, zero<half>());
    wmma::load_matrix_sync(a_frag, d_in + in_offset, 16);
    wmma::load_matrix_sync(b_frag, d_in + in_offset, 16);

    wmma::mma_sync(au_frag, a_frag, u_frag, out_frag);
    wmma::mma_sync(la_frag, l_frag, b_frag, la_frag);

    // store accumulator la_frag into shared memory and load it into
    // matrix_a
    // fragment la_mat_frag
    wmma::store_matrix_sync(la_mat_s + local_offset, la_frag, 16, wmma::mem_row_major);
    wmma::load_matrix_sync(la_mat_frag, la_mat_s + local_offset, 16);

    wmma::mma_sync(out_frag, la_mat_frag, o_frag, au_frag);

    wmma::store_matrix_sync(d_out_s + out_offset, out_frag, 16, wmma::mem_row_major);

    /* __threadfence_block(); */

    partial_sum = utils::fragment::matrix_c::last_element<WARPS_PER_BLOCK>(out_frag);
    wmma::fill_fragment(out_frag, partial_sum);
  }

  if (laneid == 0) {
    partial_sums[globalWarpIdx] = partial_sum;
  }

  this_grid().sync();

  if (laneid == 0) {
    for (size_t ii = 0; ii < globalWarpIdx; ii++) {
      adjusted_sums[localWarpIdx] += partial_sums[ii];
    }
    xprintf("------adjusted_sums[%d] = %f\n", localWarpIdx,
            (float) adjusted_sums[localWarpIdx]);
  }

  __syncwarp();
  /* __syncthreads(); */

  for (size_t ii = 0; ii < SEGMENT_SIZE; ii += WARP_SIZE) {
    const auto idx = ii + laneid;
    d_out[global_offset + idx] =
        d_out_s[d_out_s_offset + idx] + adjusted_sums[localWarpIdx];
  }
}

// SEGMENTS_PER_WARP = 1
// each warp calculates 1 segment
template <int WARPS_PER_BLOCK, int BLOCK_DIM, size_t SEGMENT_SIZE>
static __global__ void compute_wmma_prefixsum_atomic_w_atomicballot(
    const half *__restrict__ d_in, half *__restrict__ d_out,
    half *__restrict__ partial_sums, size_t num_segments, int *partial_sums_visitor) {

  __shared__ half la_mat_s[WARPS_PER_BLOCK * WMMA_TILE_SIZE];
  __shared__ half d_out_s[WARPS_PER_BLOCK * SEGMENT_SIZE];
  __shared__ half adjusted_sums[32]; // WARPS_PER_BLOCK

  const int localWarpIdx = p2_div<WARP_SIZE>(threadIdx.x);
  const int laneid       = p2_mod<WARP_SIZE>(threadIdx.x);

  const size_t globalWarpIdx = (blockIdx.x * BLOCK_DIM + threadIdx.x) / WARP_SIZE;
  const size_t global_offset = globalWarpIdx * SEGMENT_SIZE;
  const int local_offset     = localWarpIdx * WMMA_TILE_SIZE;
  const int d_out_s_offset   = localWarpIdx * SEGMENT_SIZE;
  const int num_warps        = BLOCK_DIM / WARP_SIZE;

  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> b_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> u_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> l_frag;
  wmma::fragment<wmma::matrix_b, M, N, K, half, wmma::row_major> o_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> la_frag;
  wmma::fragment<wmma::matrix_a, M, N, K, half, wmma::row_major> la_mat_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> au_frag;
  wmma::fragment<wmma::accumulator, M, N, K, half> out_frag;

  utils::fragment::matrix_b::set_upper_triangular(u_frag);
  utils::fragment::matrix_a::set_strict_lower_triangular(l_frag);

  wmma::fill_fragment(o_frag, one<half>());
  wmma::fill_fragment(out_frag, zero<half>());

  half partial_sum = zero<half>();

#pragma unroll
  for (size_t ii = 0; ii < SEGMENT_SIZE / WMMA_TILE_SIZE; ii++) {
    const int in_offset  = global_offset + ii * WMMA_TILE_SIZE;
    const int out_offset = d_out_s_offset + ii * WMMA_TILE_SIZE;

    wmma::fill_fragment(la_frag, zero<half>());
    wmma::load_matrix_sync(a_frag, d_in + in_offset, 16);
    wmma::load_matrix_sync(b_frag, d_in + in_offset, 16);

    wmma::mma_sync(au_frag, a_frag, u_frag, out_frag);
    wmma::mma_sync(la_frag, l_frag, b_frag, la_frag);

    // store accumulator la_frag into shared memory and load it into
    // matrix_a fragment la_mat_frag
    wmma::store_matrix_sync(la_mat_s + local_offset, la_frag, 16, wmma::mem_row_major);
    wmma::load_matrix_sync(la_mat_frag, la_mat_s + local_offset, 16);

    wmma::mma_sync(out_frag, la_mat_frag, o_frag, au_frag);

    wmma::store_matrix_sync(d_out_s + out_offset, out_frag, 16, wmma::mem_row_major);

    /* __threadfence_block(); */

    partial_sum = utils::fragment::matrix_c::last_element<WARPS_PER_BLOCK>(out_frag);
    wmma::fill_fragment(out_frag, partial_sum);
  }

  if (laneid == 0) {
    partial_sums[globalWarpIdx] = partial_sum;
  }

  int val = atomicAdd(partial_sums_visitor, 1);

  do {
  } while (atomicCAS(partial_sums_visitor, num_warps, 0));

  if (laneid == 0) {
    for (size_t ii = 0; ii < globalWarpIdx; ii++) {
      adjusted_sums[localWarpIdx] += partial_sums[ii];
    }
    xprintf("------adjusted_sums[%d] = %f\n", localWarpIdx,
            (float) adjusted_sums[localWarpIdx]);
  }

  __syncwarp();
  /* __syncthreads(); */

  for (size_t ii = 0; ii < SEGMENT_SIZE; ii += WARP_SIZE) {
    const auto idx = ii + laneid;
    d_out[global_offset + idx] =
        d_out_s[d_out_s_offset + idx] + adjusted_sums[localWarpIdx];
  }
}

} // namespace wmma_unsafe_prefixsum
