#pragma once

#include <stdint.h>
#include <string.h>

#include <benchmark/benchmark.h>
#include <cmath>

#if 0
#define ARGS() Args({256 * 512})->ArgNames({"num_elements"})

#define SEG_16_ARGS() Args({65535, 16})->ArgNames({"num_segments", "segment_size"})

#define SEG_32_ARGS() Args({65535, 32})->ArgNames({"num_segments", "segment_size"})

#define SEG_64_ARGS() Args({65535, 64})->ArgNames({"num_segments", "segment_size"})

#define SEG_128_ARGS() Args({65535, 128})->ArgNames({"num_segments", "segment_size"})

#define SEG_256_ARGS() Args({65535, 256})->ArgNames({"num_segments", "segment_size"})

#define SEG_512_ARGS() Args({65535, 512})->ArgNames({"num_segments", "segment_size"})

#define SEG_1024_ARGS() Args({65535, 1024})->ArgNames({"num_segments", "segment_size"})

#define SEG_2048_ARGS() Args({65535, 2048})->ArgNames({"num_segments", "segment_size"})

#define SEG_4096_ARGS() Args({64, 512*65536})->ArgNames({"num_segments", "segment_size"})

#else

static inline int64_t two_power_i(int64_t p) {
  return static_cast<int64_t>(1) << p;
}

static const int min_num_segments = 80; // number of sms

static inline void ArgsParams(benchmark::internal::Benchmark* b) {
  const int64_t min_elements_log2 = 12;
  const int64_t max_elements_log2 = 30;
  for (int64_t num_elmenets_log2 = min_elements_log2;
       num_elmenets_log2 <= max_elements_log2;
       num_elmenets_log2++) {
    const int64_t num_elements = two_power_i(num_elmenets_log2);
    b->Args({num_elements});
  }
  b->ArgNames({"num_elements"});
}

template <int BASE_SEGMENET_SIZE>
static inline void ArgsParamsSegmented(benchmark::internal::Benchmark* b) {
  const int64_t min_elements_log2 = 12;
  const int64_t max_elements_log2 = 30;
  for (int64_t num_elmenets_log2 = min_elements_log2;
       num_elmenets_log2 <= max_elements_log2;
       num_elmenets_log2++) {
    const int64_t num_elements = two_power_i(num_elmenets_log2);
    const int64_t num_segments = num_elements / BASE_SEGMENET_SIZE;
    b->Args({num_segments, BASE_SEGMENET_SIZE, num_elements});
  }
  b->ArgNames({"num_segments", "segment_size", "num_elements"});
}

#define ARGS() Apply(ArgsParams)

#define SEG_16_ARGS() Apply(ArgsParamsSegmented<16>)
#define SEG_32_ARGS() Apply(ArgsParamsSegmented<32>)
#define SEG_64_ARGS() Apply(ArgsParamsSegmented<64>)
#define SEG_128_ARGS() Apply(ArgsParamsSegmented<128>)
#define SEG_256_ARGS() Apply(ArgsParamsSegmented<256>)
#define SEG_512_ARGS() Apply(ArgsParamsSegmented<512>)
#define SEG_1024_ARGS() Apply(ArgsParamsSegmented<1024>)
#define SEG_2048_ARGS() Apply(ArgsParamsSegmented<2048>)
#define SEG_4096_ARGS() Apply(ArgsParamsSegmented<4096>)
#define SEG_8192_ARGS() Apply(ArgsParamsSegmented<8192>)
#define SEG_16384_ARGS() Apply(ArgsParamsSegmented<16384>)

// static inline bool ispow2(int x) noexcept {
//   return x > 0 && (x & (x - 1)) == 0;
// }

template <int BASE_SEGMENT_SIZE, int MAX_ELEMENTS_LOG2>
static inline void TuningArgs(benchmark::internal::Benchmark* b) {
  const int64_t num_elements          = two_power_i(MAX_ELEMENTS_LOG2);
  const int64_t segment_size_multiple = BASE_SEGMENT_SIZE;
  for (int64_t num_segments = num_elements / segment_size_multiple; num_segments >= 1;
       num_segments /= 2) {
    const int64_t segment_size = num_elements / num_segments;
    if (segment_size < BASE_SEGMENT_SIZE) {
      continue;
    }
    b->Args({num_segments, num_elements / num_segments, BASE_SEGMENT_SIZE,
             MAX_ELEMENTS_LOG2});
  }
  b->ArgNames({"num_segments", "segment_size", "base_segment_size", "max_elements_log2"});
}

// tuning where segment size multiple = 16 for elements of size upto 2^16
static inline void Tuning16_x_14(benchmark::internal::Benchmark* b) {
  TuningArgs<16, 14>(b);
}

// tuning where segment size multiple = 256 for elements of size upto 2^16
static inline void Tuning256_x_14(benchmark::internal::Benchmark* b) {
  TuningArgs<256, 14>(b);
}

// tuning where segment size multiple = 16 for elements of size upto 2^18
static inline void Tuning16_x_18(benchmark::internal::Benchmark* b) {
  TuningArgs<16, 18>(b);
}

// tuning where segment size multiple = 256 for elements of size upto 2^18
static inline void Tuning256_x_18(benchmark::internal::Benchmark* b) {
  TuningArgs<256, 18>(b);
}

// tuning where segment size multiple = 16 for elements of size upto 2^22
static inline void Tuning16_x_22(benchmark::internal::Benchmark* b) {
  TuningArgs<16, 22>(b);
}

// tuning where segment size multiple = 256 for elements of size upto 2^22
static inline void Tuning256_x_22(benchmark::internal::Benchmark* b) {
  TuningArgs<256, 22>(b);
}

// tuning where segment size multiple = 16 for elements of size upto 2^26
static inline void Tuning16_x_26(benchmark::internal::Benchmark* b) {
  TuningArgs<16, 26>(b);
}

// tuning where segment size multiple = 256 for elements of size upto 2^26
static inline void Tuning256_x_26(benchmark::internal::Benchmark* b) {
  TuningArgs<256, 26>(b);
}

// tuning where segment size multiple = 16 for elements of size upto 2^30
static inline void Tuning16_x_30(benchmark::internal::Benchmark* b) {
  TuningArgs<16, 30>(b);
}

// tuning where segment size multiple = 256 for elements of size upto 2^30
static inline void Tuning256_x_30(benchmark::internal::Benchmark* b) {
  TuningArgs<256, 30>(b);
}

#endif
