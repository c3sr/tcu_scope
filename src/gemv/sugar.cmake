# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED SRC_GEMV_SUGAR_CMAKE_)
  return()
else()
  set(SRC_GEMV_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    args.hpp
)

sugar_files(
    BENCHMARK_CUDA_SOURCES
    wmma_sharedmem.cu
    wmma_cublas.cu
    accuracy.cu
    cublas.cu
    wmma_naive.cu
)

