#pragma once

#include <cstdint>     // for uintXX_t types
#include <type_traits> // for std::is_unsigned

#include "config.hpp"

#include "cuda.hpp"
#include <cuda_fp16.h>

#ifdef __NVCC__

#ifndef STRINGIFY
#define STRINGIFY(_q) #_q
#endif

// divide by a power of 2
template <int N>
static __forceinline__ __device__ int p2_div(const int k);
template <>
__forceinline__ __device__ int p2_div<1>(const int k) {
  return k;
}
template <>
__forceinline__ __device__ int p2_div<2>(const int k) {
  return k >> 1;
}
template <>
__forceinline__ __device__ int p2_div<4>(const int k) {
  return k >> 2;
}
template <>
__forceinline__ __device__ int p2_div<8>(const int k) {
  return k >> 3;
}
template <>
__forceinline__ __device__ int p2_div<16>(const int k) {
  return k >> 4;
}
template <>
__forceinline__ __device__ int p2_div<32>(const int k) {
  return k >> 5;
}
template <>
__forceinline__ __device__ int p2_div<64>(const int k) {
  return k >> 6;
}
template <>
__forceinline__ __device__ int p2_div<128>(const int k) {
  return k >> 7;
}
template <>
__forceinline__ __device__ int p2_div<256>(const int k) {
  return k >> 8;
}
template <>
__forceinline__ __device__ int p2_div<512>(const int k) {
  return k >> 9;
}

namespace detail {
template <int x>
struct log2 {
  enum { value = 1 + log2<x / 2>::value };
};

template <>
struct log2<0> {
  enum { value = 0 };
};

template <>
struct log2<1> {
  enum { value = 1 };
};
}

// multiply by a power of 2
template <size_t N, typename T>
static __forceinline__ __device__ T p2_mul(const T k) {
  static constexpr T p2 = detail::log2<N>::value;
  if (p2 >= 8*sizeof(T)) {
      return std::numeric_limits<T>::max();
  }
  return k << p2;
}

// mod by a power of 2
template <int N>
static __forceinline__ __device__ int p2_mod(const int k) {
  static_assert(N && !(N & (N - 1)), "must be a power of two"); // is power of two
  return k & (N - 1);
}

static __forceinline__ __device__ bool is_odd(const int k) {
  return k & 1;
}

static __forceinline__ __device__ bool is_even(const int k) {
  return !is_odd(k);
}

/*
 * First, a pointer-size-related definition. Always use this as (part of) the
 * constraint string for pointer arguments to PTX asm instructions
 * (see http://docs.nvidia.com/cuda/inline-ptx-assembly/index.html#constraints)
 * it is intended to support compilation both in 64-bit and 32-bit modes.
 */

#if defined(_WIN64) || defined(__LP64__)
#define PTR_CONSTRAINT "l"
#else
#define PTR_CONSTRAINT "r"
#endif

static __global__ void convertFp32ToFp16(half *out, float *in, size_t n) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  if (idx < n) {
    out[idx] = (half) in[idx];
  }
}

template <int N>
static __global__ void setFP16(half *out, float in, size_t len) {
  size_t idx = N * (blockDim.x * blockIdx.x + threadIdx.x);
  for (size_t ii = 0; ii < N; ii++) {
    if (idx < len) {
      out[idx] = (half) in;
      idx++;
    }
  }
}

template <typename T, typename R>
void cuda_memory_set(T *mem, R val, size_t num_elements) {
  const auto block_size = 1024;
  const auto grid_size  = (num_elements + block_size - 1) / block_size;
  if (grid_size < CUDA_MAX_GRID_SIZE) {
    setFP16<1><<<grid_size, block_size>>>(mem, val, num_elements);
  }
  const auto coarsening_factor = grid_size / CUDA_MAX_GRID_SIZE + 1;
  switch (coarsening_factor) {
#define dispatch_mem_set(n)                                                              \
  case n:                                                                                \
    setFP16<n><<<grid_size, block_size>>>(mem, val, num_elements);                       \
    return
    dispatch_mem_set(1);
    dispatch_mem_set(2);
    dispatch_mem_set(3);
    dispatch_mem_set(4);
    dispatch_mem_set(5);
    dispatch_mem_set(6);
    dispatch_mem_set(7);
    dispatch_mem_set(8);
    dispatch_mem_set(9);
    dispatch_mem_set(10);
    dispatch_mem_set(11);
    dispatch_mem_set(12);
    dispatch_mem_set(13);
    dispatch_mem_set(14);
    dispatch_mem_set(15);
    dispatch_mem_set(16);
    dispatch_mem_set(17);
    dispatch_mem_set(18);
    dispatch_mem_set(19);
    dispatch_mem_set(20);
    dispatch_mem_set(21);
    dispatch_mem_set(22);
    dispatch_mem_set(23);
    dispatch_mem_set(24);
    dispatch_mem_set(25);
    dispatch_mem_set(26);
    dispatch_mem_set(27);
    dispatch_mem_set(28);
    dispatch_mem_set(29);
    dispatch_mem_set(30);
    dispatch_mem_set(31);
    dispatch_mem_set(32);
#undef dispatch_mem_set
  }
}

template <typename T>
static __forceinline__ __device__ constexpr T one() {
  return T(1);
}

template <>
__forceinline__ __device__ half one<half>() {
  return __half_raw{.x = 0x3c00};
}

namespace half_constants {
__forceinline__ __device__ half nan() {
  return __half_raw{.x = 0x7fff};
}

__forceinline__ __device__ half inf() {
  return __half_raw{.x = 0x7c00};
}

__forceinline__ __device__ half ninf() {
  return __half_raw{.x = 0x7c00};
}

__forceinline__ __device__ half epsilon() {
  return __half_raw{.x = 0x1000};
}
} // namespace half_constants

template <typename T>
static __forceinline__ __device__ constexpr T zero() {
  return T(0);
}

template <>
__forceinline__ __device__ half zero<half>() {
  return __half_raw{.x = 0};
}

#ifndef __NV_STD_MIN
#define __NV_STD_MIN(a, b) (((b) < (a)) ? (b) : (a))
#endif

template <class Dest, class Src>
static __device__ void bit_cast(const Dest &d, const Src &x) {
  memset(d, 0, sizeof(Dest));
  memcpy(d, x, __NV_STD_MIN(sizeof(Dest), sizeof(Src)));
}

#define DEFINE_SPECIAL_REGISTER_GETTER(special_register_name, value_type, width_in_bits) \
  __forceinline__ __device__ value_type special_register_name() {                        \
    value_type ret;                                                                      \
    if (std::is_unsigned<value_type>::value) {                                           \
      asm("mov.u" STRINGIFY(width_in_bits) " %0, %" STRINGIFY(special_register_name) ";" \
          : "=r"(ret));                                                                  \
    } else {                                                                             \
      asm("mov.s" STRINGIFY(width_in_bits) " %0, %" STRINGIFY(special_register_name) ";" \
          : "=r"(ret));                                                                  \
    }                                                                                    \
    return ret;                                                                          \
  }

DEFINE_SPECIAL_REGISTER_GETTER(laneid, uint32_t, 32);
DEFINE_SPECIAL_REGISTER_GETTER(warpid, uint32_t, 32);
DEFINE_SPECIAL_REGISTER_GETTER(smid, uint32_t, 32);
DEFINE_SPECIAL_REGISTER_GETTER(nsmid, uint32_t, 32);
DEFINE_SPECIAL_REGISTER_GETTER(lanemask_lt, uint32_t, 32);
DEFINE_SPECIAL_REGISTER_GETTER(lanemask_le, uint32_t, 32);
DEFINE_SPECIAL_REGISTER_GETTER(lanemask_eq, uint32_t, 32);
DEFINE_SPECIAL_REGISTER_GETTER(lanemask_ge, uint32_t, 32);
DEFINE_SPECIAL_REGISTER_GETTER(lanemask_gt, uint32_t, 32);
DEFINE_SPECIAL_REGISTER_GETTER(dynamic_smem_size, uint32_t, 32);
DEFINE_SPECIAL_REGISTER_GETTER(total_smem_size, uint32_t, 32);

#undef DEFINE_SPECIAL_REGISTER_GETTER

template <typename T>
__forceinline__ __device__ T ldg(const T *ptr) {
#if __CUDA_ARCH__ >= 320
  return __ldg(ptr);
#else
  return *ptr; // maybe we should ld.cg or ld.cs here?
#endif
}

/**
 * See @link
 * http://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-isspacep
 */
#define DEFINE_IS_IN_MEMORY_SPACE(_which_space)                                          \
  __forceinline__ __device__ int is_in_##_which_space##_memory(const void *ptr) {        \
    int result;                                                                          \
    asm("{"                                                                              \
        ".reg .pred p;\n\t"                                                              \
        "isspacep." STRINGIFY(_which_space) " p, %1;\n\t"                                \
                                            "selp.b32 %0, 1, 0, p;\n\t"                  \
                                            "}"                                          \
        : "=r"(result)                                                                   \
        : PTR_CONSTRAINT(ptr));                                                          \
    return result;                                                                       \
  }

DEFINE_IS_IN_MEMORY_SPACE(const)
DEFINE_IS_IN_MEMORY_SPACE(global)
DEFINE_IS_IN_MEMORY_SPACE(local)
DEFINE_IS_IN_MEMORY_SPACE(shared)

#undef DEFINE_IS_IN_MEMORY_SPACE

#undef PTR_CONSTRAINT

#undef STRINGIFY

#endif /* __NVCC__ */
