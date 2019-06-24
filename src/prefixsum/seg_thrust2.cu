
#include <benchmark/benchmark.h>

#include "init/init.hpp"
#include "prefixsum/args.hpp"
#include "utils/utils.hpp"

#include <thrust/execution_policy.h>
#include <thrust/scan.h>

static void THRUST_SEGMENTED2_PREFIXSUM(benchmark::State &state) {
  const size_t num_segments = state.range(0);
  const size_t segment_size = state.range(1);
  const size_t num_elements = num_segments * segment_size;

  cudaEvent_t start, stop;
  half *d_in_fp16 = nullptr;
  half *d_out     = nullptr;

  try {
    PRINT_IF_ERROR(cudaMalloc(&d_in_fp16, num_elements * sizeof(half)));
    PRINT_IF_ERROR(cudaMalloc(&d_out, num_elements * sizeof(half)));

    cuda_memory_set(d_in_fp16, 0.001f, num_elements);

    PRINT_IF_ERROR(cudaDeviceSynchronize());

    PRINT_IF_ERROR(cudaEventCreate(&start));
    PRINT_IF_ERROR(cudaEventCreate(&stop));

    defer(cudaEventDestroy(start));
    defer(cudaEventDestroy(stop));

    for (auto _ : state) {
      PRINT_IF_ERROR(cudaEventRecord(start));

      for (size_t ii = 0; ii < num_segments; ii++) {
        thrust::inclusive_scan(thrust::device, d_in_fp16 + ii * segment_size,
                               d_in_fp16 + (ii + 1) * segment_size,
                               d_out + ii * segment_size);
      }

      PRINT_IF_ERROR(cudaEventRecord(stop));
      PRINT_IF_ERROR(cudaEventSynchronize(stop));

      state.PauseTiming();

      float msecTotal = 0.0f;
      PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
      state.SetIterationTime(msecTotal / 1000);
      state.ResumeTiming();
    }

    state.counters.insert({{"num_segments", num_segments},
                           {"num_elements", num_segments * segment_size},
                           {"segment_size", segment_size},
                           {"flops",
                            {state.iterations() * 1.0 * num_elements,
                             benchmark::Counter::kAvgThreadsRate}}});

#if 0
  half *h_out = new half[num_elements];
  PRINT_IF_ERROR(cudaMemcpy(h_out, d_out, num_elements * sizeof(half), cudaMemcpyDeviceToHost));

  int errors        = 0;
  float correct_sum = 0;
  for (int i = 0; i < num_elements; i++) {
    correct_sum += h_in[i];
    if (fabs(half_to_float((h_out[i])) - correct_sum) > 0.01) {
      errors++;
      printf("Expected %f, get h_out[%d] = %f\n", correct_sum, i, half_to_float(h_out[i]));
    }
  }

  if (errors > 0) {
    printf("THRUST_SEGMENTED2_PREFIXSUM does not agree with SEQUENTIAL! %d errors!\n", errors);
  } else {
    printf("Results verified: they agree.\n\n");
  }

  delete h_out;
#endif

    cudaFree(d_in_fp16);
  } catch (...) {
    cudaFree(d_in_fp16);

    cudaDeviceReset();
    const auto p = std::current_exception();
    std::rethrow_exception(p);
  }
}

BENCHMARK(THRUST_SEGMENTED2_PREFIXSUM)->SEG_16_ARGS()->UseManualTime();
BENCHMARK(THRUST_SEGMENTED2_PREFIXSUM)->SEG_32_ARGS()->UseManualTime();
BENCHMARK(THRUST_SEGMENTED2_PREFIXSUM)->SEG_64_ARGS()->UseManualTime();
BENCHMARK(THRUST_SEGMENTED2_PREFIXSUM)->SEG_128_ARGS()->UseManualTime();
BENCHMARK(THRUST_SEGMENTED2_PREFIXSUM)->SEG_256_ARGS()->UseManualTime();
BENCHMARK(THRUST_SEGMENTED2_PREFIXSUM)->SEG_512_ARGS()->UseManualTime();
BENCHMARK(THRUST_SEGMENTED2_PREFIXSUM)->SEG_1024_ARGS()->UseManualTime();
BENCHMARK(THRUST_SEGMENTED2_PREFIXSUM)->SEG_2048_ARGS()->UseManualTime();
BENCHMARK(THRUST_SEGMENTED2_PREFIXSUM)->SEG_4096_ARGS()->UseManualTime();
BENCHMARK(THRUST_SEGMENTED2_PREFIXSUM)->SEG_8192_ARGS()->UseManualTime();
BENCHMARK(THRUST_SEGMENTED2_PREFIXSUM)->SEG_16384_ARGS()->UseManualTime();

static void THRUST_TUNE_SEGMENTED2_PREFIXSUM(benchmark::State &state) {
  THRUST_SEGMENTED2_PREFIXSUM(state);
}

#define RUN_THRUST_TUNE(TUNE_ARGS)                                                       \
  BENCHMARK(THRUST_TUNE_SEGMENTED2_PREFIXSUM)->Apply(TUNE_ARGS)->UseManualTime();

// RUN_THRUST_TUNE(Tuning16_x_14);
// RUN_THRUST_TUNE(Tuning16_x_18);
RUN_THRUST_TUNE(Tuning16_x_22);
// RUN_THRUST_TUNE(Tuning16_x_26);
RUN_THRUST_TUNE(Tuning16_x_30);
