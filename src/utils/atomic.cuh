#pragma once

#include <cstdint>     // for uintXX_t types
#include <type_traits> // for std::is_unsigned

#include "cuda.hpp"
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>

#ifdef __NVCC__

#ifndef __HALF_TO_US
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#endif

// from
// https://github.com/torch/cutorch/blob/653811fa40a780dfa8f0e110f9febf5dfed8f0f0/lib/THC/THCAtomics.cuh#L97
#if 0
static inline __device__ void atomicAdd(half *address, half val) {
  unsigned int *address_as_ui =
      (unsigned int *) ((char *) address - ((size_t) address & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;

  do {
    assumed = old;
    __half_raw hsum;
    hsum.x      = (size_t) address & 2 ? (old >> 16) : (old & 0xffff);
    half tmpres = __hadd(half(hsum), val);
    hsum        = __half_raw(tmpres);
    old         = (size_t) address & 2 ? (old & 0xffff) | (hsum.x << 16)
                               : (old & 0xffff0000) | hsum.x;
    old = atomicCAS(address_as_ui, assumed, old);
  } while (assumed != old);
}
#endif

static inline __device__ short to_short(half x) {
  return __HALF_TO_US(x);
}

#endif /* __NVCC__ */
