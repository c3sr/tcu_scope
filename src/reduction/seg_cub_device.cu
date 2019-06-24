#define CUB_HALF_OPTIMIZATION 1

#include <benchmark/benchmark.h>

#include "init/init.hpp"
#include "reduction/args.hpp"
#include "utils/utils.hpp"

#include <cub/cub.cuh>

using namespace cub;

static void CUB_DEVICE_SEGMENTED_REDUCTION(benchmark::State &state) {
  const size_t num_segments = state.range(0);
  const size_t segment_size = state.range(1);
  const size_t num_elements = num_segments * segment_size;

  int *h_offsets = new int[num_segments + 1];

  for (int i = 0; i < num_segments + 1; i++) {
    h_offsets[i] = i * segment_size;
  }

  int *d_offsets  = nullptr;
  half *d_in_fp16 = nullptr;
  half *d_out     = nullptr;
  cudaEvent_t start, stop;

  void *d_temp_storage      = NULL;
  size_t temp_storage_bytes = 0;

  defer(cudaDeviceReset());

  try {
    PRINT_IF_ERROR(cudaMalloc(&d_offsets, (num_segments + 1) * sizeof(int)));
    PRINT_IF_ERROR(cudaMalloc(&d_in_fp16, num_elements * sizeof(half)));
    PRINT_IF_ERROR(cudaMalloc(&d_out, num_segments * sizeof(half)));

    PRINT_IF_ERROR(cudaMemcpy(d_offsets, h_offsets, (num_segments + 1) * sizeof(int),
                              cudaMemcpyHostToDevice));

    cuda_memory_set(d_in_fp16, 0.001f, num_elements);

    PRINT_IF_ERROR(cudaDeviceSynchronize());

    PRINT_IF_ERROR(cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
                                                   d_in_fp16, d_out, num_segments,
                                                   d_offsets, d_offsets + 1));

    PRINT_IF_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

    PRINT_IF_ERROR(cudaEventCreate(&start));
    PRINT_IF_ERROR(cudaEventCreate(&stop));

    defer(cudaEventDestroy(start));
    defer(cudaEventDestroy(stop));

    for (auto _ : state) {
      PRINT_IF_ERROR(cudaEventRecord(start));

      PRINT_IF_ERROR(cub::DeviceSegmentedReduce::Sum(d_temp_storage, temp_storage_bytes,
                                                     d_in_fp16, d_out, num_segments,
                                                     d_offsets, d_offsets + 1));

      PRINT_IF_ERROR(cudaEventRecord(stop));
      PRINT_IF_ERROR(cudaEventSynchronize(stop));

      state.PauseTiming();

      float msecTotal = 0.0f;
      PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
      state.SetIterationTime(msecTotal / 1000);
      state.ResumeTiming();
    }

    state.counters.insert({{"num_segments", num_segments},
                           {"segment_size", segment_size},
                           {"num_elements", num_segments * segment_size},
                           {"flops",
                            {state.iterations() * 1.0 * num_segments * segment_size,
                             benchmark::Counter::kAvgThreadsRate}}});

#if 0
  half *h_out = new half[num_segments];
  PRINT_IF_ERROR(cudaMemcpy(h_out, d_out, num_segments * sizeof(half),
                            cudaMemcpyDeviceToHost));

  float correct_segment_sum = 0;
  for (int i = 0; i < segment_size; i++) {
    correct_segment_sum += h_in[i];
  }

  int errors = 0;
  for (int i = 0; i < num_segments; i++) {
    if (fabs(half_to_float(h_out[i]) - correct_segment_sum) > 0.001) {
      errors++;
      if (errors < 10) {
        printf("segment %d has sum %f (expected %f)\n", i,
               half_to_float(h_out[i]), correct_segment_sum);
      }
    }
  }

  if (errors > 0) {
    printf("CUB_DEVICE_SEGMENTED_REDUCTION does not agree with SEQUENTIAL! %d "
           "errors!\n",
           errors);
  } else {
    printf("Results verified: they agree.\n\n");
  }

  delete h_out;
#endif
    delete[] h_offsets;
    cudaFree(d_offsets);
    cudaFree(d_in_fp16);
    cudaFree(d_out);
    cudaFree(d_temp_storage);
  } catch (...) {
    delete[] h_offsets;
    cudaFree(d_offsets);
    cudaFree(d_in_fp16);
    cudaFree(d_out);
    cudaFree(d_temp_storage);

    cudaDeviceReset();
    const auto p = std::current_exception();
    std::rethrow_exception(p);
  }
}

BENCHMARK(CUB_DEVICE_SEGMENTED_REDUCTION)->SEG_16_ARGS()->UseManualTime();
BENCHMARK(CUB_DEVICE_SEGMENTED_REDUCTION)->SEG_32_ARGS()->UseManualTime();
BENCHMARK(CUB_DEVICE_SEGMENTED_REDUCTION)->SEG_64_ARGS()->UseManualTime();
BENCHMARK(CUB_DEVICE_SEGMENTED_REDUCTION)->SEG_128_ARGS()->UseManualTime();
BENCHMARK(CUB_DEVICE_SEGMENTED_REDUCTION)->SEG_256_ARGS()->UseManualTime();
BENCHMARK(CUB_DEVICE_SEGMENTED_REDUCTION)->SEG_512_ARGS()->UseManualTime();
BENCHMARK(CUB_DEVICE_SEGMENTED_REDUCTION)->SEG_1024_ARGS()->UseManualTime();
BENCHMARK(CUB_DEVICE_SEGMENTED_REDUCTION)->SEG_2048_ARGS()->UseManualTime();
BENCHMARK(CUB_DEVICE_SEGMENTED_REDUCTION)->SEG_4096_ARGS()->UseManualTime();
BENCHMARK(CUB_DEVICE_SEGMENTED_REDUCTION)->SEG_8192_ARGS()->UseManualTime();
BENCHMARK(CUB_DEVICE_SEGMENTED_REDUCTION)->SEG_16384_ARGS()->UseManualTime();

static void CUB_DEVICE_TUNE_SEGMENTED_REDUCTION(benchmark::State &state) {
  CUB_DEVICE_SEGMENTED_REDUCTION(state);
}

#define RUN_CUB_DEVICE_TUNE(TUNE_ARGS)                                                   \
  BENCHMARK(CUB_DEVICE_TUNE_SEGMENTED_REDUCTION)->Apply(TUNE_ARGS)->UseManualTime();

// RUN_CUB_DEVICE_TUNE(Tuning16_x_14);
// RUN_CUB_DEVICE_TUNE(Tuning16_x_18);
RUN_CUB_DEVICE_TUNE(Tuning16_x_22);
// RUN_CUB_DEVICE_TUNE(Tuning16_x_26);
RUN_CUB_DEVICE_TUNE(Tuning16_x_30);
