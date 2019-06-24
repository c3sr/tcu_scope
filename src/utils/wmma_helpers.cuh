#pragma once

#include <cstdint>     // for uintXX_t types
#include <type_traits> // for std::is_unsigned

#include "cuda.hpp"
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>

#include "utils/cuda_helpers.cuh"

#ifdef __NVCC__
#ifndef WARP_SIZE
#define WARP_SIZE (32)
#endif // WARP_SIZE

namespace utils {
namespace fragment {

  template <int N>
  static __forceinline__ __device__ int get_bit(int val) {
    return (val & (1 << N)) >> N;
  }
  namespace matrix_a {

#if 0
    template <typename F>
    static __forceinline__ __device__ int for_each_offset() {
      const int tid = threadIdx.x;
      const int a   = p2_mod<8>(tid);
      int offset;
      if (a < 4) {
        offset = (tid < 16) ? 0 : 4;
      } else {
        offset = (tid < 16) ? 8 : 12;
      }
      const int b = p2_mod<4>(tid);
      offset      = offset + 16 * b;
#pragma unroll
      for (int ii = 0; ii < 4; ii++) {
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
          const int idx = offset + 64 * ii + jj;
          F(idx);
        }
      }
    }

    template <typename F>
    static __forceinline__ __device__ int for_each_index() {
      const int tid = threadIdx.x;
      const int a   = p2_mod<8>(tid);
      int offset;
      if (a < 4) {
        offset = (tid < 16) ? 0 : 4;
      } else {
        offset = (tid < 16) ? 8 : 12;
      }
      const int b = p2_mod<4>(tid);
      offset      = offset + 16 * b;
#pragma unroll
      for (int ii = 0; ii < 4; ii++) {
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
          const int idx = offset + 64 * ii + jj;
          F(p2_div<16>(idx), p2_mod<16>(idx));
        }
      }
    }
#endif
    template <typename T>
    static __forceinline__ __device__ void set_upper_triangular(T &k) {
      const int tid = p2_mod<WARP_SIZE>(threadIdx.x);
      const int a   = p2_mod<8>(tid);
      int offset;
      if (a < 4) {
        offset = (tid < 16) ? 0 : 4;
      } else {
        offset = (tid < 16) ? 8 : 12;
      }
      const int b = p2_mod<4>(tid);
      offset      = offset + 16 * b;
#pragma unroll
      for (int ii = 0; ii < 4; ii++) {
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
          const int idx     = offset + 64 * ii + jj;
          const int row     = p2_div<16>(idx);
          const int col     = p2_mod<16>(idx);
          const int _offset = (ii * 4 + jj);
          k.x[_offset]      = row >= col ? zero<half>() : one<half>();
        }
      }
    }
    template <typename T>
    static __forceinline__ __device__ void set_strict_upper_triangular(T &k) {
      const int tid = p2_mod<WARP_SIZE>(threadIdx.x);
      const int a   = p2_mod<8>(tid);
      int offset;
      if (a < 4) {
        offset = (tid < 16) ? 0 : 4;
      } else {
        offset = (tid < 16) ? 8 : 12;
      }
      const int b = p2_mod<4>(tid);
      offset      = offset + 16 * b;
#pragma unroll
      for (int ii = 0; ii < 4; ii++) {
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
          const int idx     = offset + 64 * ii + jj;
          const int row     = p2_div<16>(idx);
          const int col     = p2_mod<16>(idx);
          const int _offset = (ii * 4 + jj);
          k.x[_offset]      = row > col ? zero<half>() : one<half>();
        }
      }
    }

    template <typename T>
    static __forceinline__ __device__ void set_lower_triangular(T &k) {
      const int tid = p2_mod<WARP_SIZE>(threadIdx.x);
      const int a   = p2_mod<8>(tid);
      int offset;
      if (a < 4) {
        offset = (tid < 16) ? 0 : 4;
      } else {
        offset = (tid < 16) ? 8 : 12;
      }
      const int b = p2_mod<4>(tid);
      offset      = offset + 16 * b;
#pragma unroll
      for (int ii = 0; ii < 4; ii++) {
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
          const int idx     = offset + 64 * ii + jj;
          const int row     = p2_div<16>(idx);
          const int col     = p2_mod<16>(idx);
          const int _offset = (ii * 4 + jj);
          k.x[_offset]      = row <= col ? zero<half>() : one<half>();
        }
      }
    }
    template <typename T>
    static __forceinline__ __device__ void set_strict_lower_triangular(T &k) {
      const int tid = p2_mod<WARP_SIZE>(threadIdx.x);
      const int a   = p2_mod<8>(tid);
      int offset;
      if (a < 4) {
        offset = (tid < 16) ? 0 : 4;
      } else {
        offset = (tid < 16) ? 8 : 12;
      }
      const int b = p2_mod<4>(tid);
      offset      = offset + 16 * b;
#pragma unroll
      for (int ii = 0; ii < 4; ii++) {
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
          const int idx     = offset + 64 * ii + jj;
          const int row     = p2_div<16>(idx);
          const int col     = p2_mod<16>(idx);
          const int _offset = (ii * 4 + jj);
          k.x[_offset]      = row < col ? zero<half>() : one<half>();
        }
      }
    }

    template <typename T>
    static __forceinline__ __device__ void set_first_row_ones(T &k) {
      const int tid = p2_mod<WARP_SIZE>(threadIdx.x);
      const int a   = p2_mod<8>(tid);
      int offset;
      if (a < 4) {
        offset = (tid < 16) ? 0 : 4;
      } else {
        offset = (tid < 16) ? 8 : 12;
      }
      const int b = p2_mod<4>(tid);
      offset      = offset + 16 * b;
#pragma unroll
      for (int ii = 0; ii < 4; ii++) {
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
          const int idx     = offset + 64 * ii + jj;
          const int row     = p2_div<16>(idx);
          const int _offset = (ii * 4 + jj);
          k.x[_offset]      = row == 0 ? one<half>() : zero<half>();
        }
      }
    }
    template <typename T>
    static __forceinline__ __device__ void set_first_column_ones(T &k) {
      const int tid = p2_mod<WARP_SIZE>(threadIdx.x);
      const int a   = p2_mod<8>(tid);
      int offset;
      if (a < 4) {
        offset = (tid < 16) ? 0 : 4;
      } else {
        offset = (tid < 16) ? 8 : 12;
      }
      const int b = p2_mod<4>(tid);
      offset      = offset + 16 * b;
#pragma unroll
      for (int ii = 0; ii < 4; ii++) {
#pragma unroll
        for (int jj = 0; jj < 4; jj++) {
          const int idx     = offset + 64 * ii + jj;
          const int col     = p2_mod<16>(idx);
          const int _offset = (ii * 4 + jj);
          k.x[_offset]      = col == 0 ? one<half>() : zero<half>();
        }
      }
    }

  } // namespace matrix_a
  namespace matrix_b {
    static __forceinline__ __device__ int get_row() {
      const int tid = p2_mod<WARP_SIZE>(threadIdx.x);
      // permute[{a4_, a3_, a2_, a1_, a0_}] := {a3, a4, 0, a1, a0};
      // permute[int_Integer]
      //   := int->FromDigits[permute[PadLeft[IntegerDigits[int, 2], 5]], 2];
      /////////////////////////
      // note you might be able to do the bitreverse bellow using __byte_perm , but
      // __byte_perm works at the byte level
      // const int row = (((get_bit<3>(tid) << 4) + get_bit<4>(tid)) << 3) +
      // p2_mod<4>(tid);
      const int row = (tid & 0x10) >> 2 + tid & 0xB;
      return row;
    }

    template <typename T>
    static __forceinline__ __device__ void set_upper_triangular(T &k) {
      const int row = get_row();
#pragma unroll
      for (int ii = 0; ii < k.num_elements; ii++) {
        k.x[ii] = row < ii ? zero<half>() : one<half>();
      }
    }
    template <typename T>
    static __forceinline__ __device__ void set_strict_upper_triangular(T &k) {
      const int row = get_row();
#pragma unroll
      for (int ii = 0; ii < k.num_elements; ii++) {
        k.x[ii] = row <= ii ? zero<half>() : one<half>();
      }
    }

    template <typename T>
    static __forceinline__ __device__ void set_lower_triangular(T &k) {
      const int row = get_row();
#pragma unroll
      for (int ii = 0; ii < k.num_elements; ii++) {
        k.x[ii] = row > ii ? zero<half>() : one<half>();
      }
    }
    template <typename T>
    static __forceinline__ __device__ void set_strict_lower_triangular(T &k) {
      const int row = get_row();
#pragma unroll
      for (int ii = 0; ii < k.num_elements; ii++) {
        k.x[ii] = row >= ii ? zero<half>() : one<half>();
      }
    }

    template <typename T>
    static __forceinline__ __device__ void set_first_row_ones(T &k) {
      const int row = get_row();
      if (row == 0) {
#pragma unroll
        for (int ii = 0; ii < k.num_elements; ii++) {
          k.x[ii] = one<half>();
        }
      } else {
#pragma unroll
        for (int ii = 0; ii < k.num_elements; ii++) {
          k.x[ii] = zero<half>();
        }
      }
    }
    template <typename T>
    static __forceinline__ __device__ void set_first_column_ones(T &k) {
      const int row = get_row();
#pragma unroll
      for (int ii = 0; ii < k.num_elements; ii++) {
        k.x[ii] = ii == 0 ? one<half>() : zero<half>();
      }
    }
  } // namespace matrix_b

  namespace matrix_c {
    template <typename T>
    static __forceinline__ __device__ half first_element(T &k) {
      const int tid = p2_mod<WARP_SIZE>(threadIdx.x);
      if (laneid == 0) {
        return k.x[0];
      }
      return zero<half>(); // undefined
    }

    namespace detail {
      constexpr __host__ __device__ int min_blocksize(size_t x, size_t y) {
        return x <= y ? x : y;
      }

      // Returns the most significant bit of a 32 bit number
      static __forceinline__ __device__ int __bfind(unsigned i) {
        int b;
        asm volatile("bfind.u32 %0, %1;" : "=r"(b) : "r"(i));
        return b;
      }
    } // namespace detail

    template <int WARPS_PER_BLOCK, typename T>
    static __forceinline__ __device__ half last_element_smem(T &k) {
      static constexpr int smem_length = detail::min_blocksize(WARPS_PER_BLOCK, 16);
      __shared__ half last_element[smem_length];
      const int warpid = p2_div<WARP_SIZE>(threadIdx.x);
      const int laneid = p2_mod<WARP_SIZE>(threadIdx.x);
      if (laneid == (WARP_SIZE - 1)) {
        last_element[warpid] = k.x[k.num_elements - 1];
      }
      __syncwarp();
      return last_element[warpid];
    }
    template <int WARPS_PER_BLOCK, typename T>
    static __forceinline__ __device__ half last_element_pred(T &k) {
      const int laneid     = p2_mod<WARP_SIZE>(threadIdx.x);
      const bool predicate = laneid == (WARP_SIZE - 1);
      const int vote       = __ballot_sync(0xFFFFFFFF, predicate);
      half last_element;
      if (predicate) {
        last_element = k.x[k.num_elements - 1];
      }
      if (vote) {
        last_element = __shfl_sync(0xFFFFFFFF, last_element, detail::__bfind(vote));
      }
      return last_element;
    }
    template <int WARPS_PER_BLOCK, typename T>
    static __forceinline__ __device__ half last_element(T &k) {
      const int laneid     = p2_mod<WARP_SIZE>(threadIdx.x);
      const bool predicate = laneid == (WARP_SIZE - 1);
      half last_element;
      if (predicate) {
        last_element = k.x[k.num_elements - 1];
      }
      last_element = __shfl_sync(0xFFFFFFFF, last_element, WARP_SIZE - 1);
      return last_element;
    }

    template <typename T>
    static __forceinline__ __device__ void store_first_element(half *out, T &k) {
      const int tid = p2_mod<WARP_SIZE>(threadIdx.x);
      if (tid == 0) {
        out[0] = k.x[0];
      }
    }

    template <typename T>
    static __forceinline__ __device__ void store_first_row(half *out, T &k) {
      const int tid = p2_mod<WARP_SIZE>(threadIdx.x);
      const int row = 8 * (p2_mod<16>(tid) > 8) + 2 * (p2_mod<4>(tid) > 2);
      if (row == 0) {
        const int col = 4 * (tid > 16) + 8 * (p2_mod<8>(tid) > 4) + p2_mod<2>(tid);
        out[col]      = k.x[0];
        out[col + 2]  = k.x[1];
      }
    }

    template <typename T>
    static __forceinline__ __device__ void store_first_column(half *out, T &k) {
      const int tid = p2_mod<WARP_SIZE>(threadIdx.x);
      const int col = 4 * (tid > 16) + 8 * (p2_mod<8>(tid) > 4) + p2_mod<2>(tid);
      if (col == 0) {
        const int row = 8 * (p2_mod<16>(tid) > 8) + 2 * (p2_mod<4>(tid) > 2);
        out[row]      = k.x[0];
        out[row + 1]  = k.x[2];
        out[row + 4]  = k.x[5];
        out[row + 5]  = k.x[7];
      }
    }

  } // namespace matrix_c

} // namespace fragment
} // namespace utils

#endif /* __NVCC__ */
