# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED SRC_UTILS_SUGAR_CMAKE_)
  return()
else()
  set(SRC_UTILS_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    mpl.hpp
    hostname.hpp
    cuda.hpp
    debug.hpp
    fp16_conversion.hpp
    error.hpp
    unused.hpp
    poly.hpp
    timer.hpp
    cublas.hpp
    benchmark.hpp
    curand.hpp
    nocopy.hpp
    compat.hpp
    utils.hpp
    errchk.hpp
    defer.hpp
    commandlineflags.hpp
    backward.hpp
)

sugar_files(
    BENCHMARK_CUDA_HEADERS
    atomic.cuh
    cuda_helpers.cuh
    wmma_helpers.cuh
)

sugar_files(
    BENCHMARK_SOURCES
    commandlineflags.cpp
    backward.cpp
)

