# This file generated automatically by:
#   generate_sugar_files.py
# see wiki for more info:
#   https://github.com/ruslo/sugar/wiki/Collecting-sources

if(DEFINED SRC_INIT_SUGAR_CMAKE_)
  return()
else()
  set(SRC_INIT_SUGAR_CMAKE_ 1)
endif()

include(sugar_files)

sugar_files(
    BENCHMARK_HEADERS
    init.hpp
    logger.hpp
    cuda.hpp
    cublas.hpp
    flags.hpp
)

sugar_files(
    BENCHMARK_SOURCES
    flags.cpp
    cublas.cpp
    cuda.cpp
    logger.cpp
    init.cpp
)

