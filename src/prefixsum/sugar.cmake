# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED SRC_PREFIXSUM_SUGAR_CMAKE_)
  return()
else()
  set(SRC_PREFIXSUM_SUGAR_CMAKE_ 1)
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
    seg_cub_block.cu
    seg_wmma_16n_opt.cu
    seg_cub_warp.cu
    wmma_cg.cu
    seg_wmma_16.cu
    cub.cu
    wmma_atomic.cu
    seg_cub_warp_original.cu
    seg_thrust.cu
    thrust.cu
    seg_wmma_256n_block.cu
    seg_wmma_256.cu
    seg_wmma_16n_block.cu
    seg_thrust2.cu
    seg_cub_block_original.cu
    seg_wmma_256n.cu
    seg_wmma_16n.cu
    wmma_3kers_block.cu
    cub_original.cu
    wmma_3kers.cu
)

