# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED SRC_GEMM_SUGAR_CMAKE_)
  return()
else()
  set(SRC_GEMM_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    utils.hpp
    args.hpp
)

sugar_files(
    BENCHMARK_CUDA_SOURCES
    wmma_sharedmem.cu
    accuracy.cu
    wmma.cu
    cutlass.cu
    cublas.cu
    wmma_naive.cu
)

