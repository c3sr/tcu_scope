# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED SRC_UNSAFE_REDUCTION_SUGAR_CMAKE_)
  return()
else()
  set(SRC_UNSAFE_REDUCTION_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    args.hpp
)

sugar_files(
    BENCHMARK_CUDA_HEADERS
    kernel.cuh
)

sugar_files(
    BENCHMARK_CUDA_SOURCES
    seg_wmma_16n_opt.cu
    wmma_cg.cu
    seg_wmma_16.cu
    wmma_atomic.cu
    wmma_2kers.cu
    seg_wmma_256n_block.cu
    seg_wmma_256.cu
    seg_wmma_16n_block.cu
    wmma_2kers_block.cu
    seg_wmma_256n.cu
    seg_wmma_16n.cu
)

