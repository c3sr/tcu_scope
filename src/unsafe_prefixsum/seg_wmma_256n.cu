
#include <benchmark/benchmark.h>

#include "init/init.hpp"
#include "prefixsum/args.hpp"
#include "utils/utils.hpp"

#include "kernel.cuh"

using namespace wmma_unsafe_prefixsum;

template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK>
void tryCUDA_UNSAFE_WMMA_SEGMENTED_PREFIXSUM_256N(benchmark::State &state) {
  const size_t num_segments = state.range(0);
  const size_t segment_size = state.range(1);

  if (segment_size != SEGMENT_SIZE) {
    state.SkipWithError(fmt::format("segment_size={} must be equal to SEGMENT_SIZE={} ",
                                    segment_size, SEGMENT_SIZE)
                            .c_str());
    return;
  }

  if (segment_size % WMMA_TILE_SIZE != 0) {
    state.SkipWithError("segment_size must be multiples of WMMA_TILE_SIZE");
    return;
  }

  const int BLOCK_DIM       = WARPS_PER_BLOCK * WARP_SIZE;
  const size_t num_elements = num_segments * segment_size;

  half *d_in_fp16;
  half *d_out;

  try {
    PRINT_IF_ERROR(cudaMalloc(&d_in_fp16, num_elements * sizeof(half)));
    PRINT_IF_ERROR(cudaMalloc(&d_out, num_elements * sizeof(half)));

    cuda_memory_set(d_in_fp16, 0.001f, num_elements);

    dim3 gridDim, blockDim;
    blockDim.x = BLOCK_DIM;
    gridDim.x  = (num_segments + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

    if (gridDim.x >= CUDA_MAX_GRID_SIZE) {
      state.SkipWithError(
          fmt::format("gridDim.x={} is greater than CUDA_MAX_GRID_SIZE", gridDim.x)
              .c_str());
      return;
    }

    cudaEvent_t start, stop;
    PRINT_IF_ERROR(cudaEventCreate(&start));
    PRINT_IF_ERROR(cudaEventCreate(&stop));

    defer(cudaEventDestroy(start));
    defer(cudaEventDestroy(stop));

    for (auto _ : state) {
      PRINT_IF_ERROR(cudaEventRecord(start));

      compute_wmma_segmented_prefixsum_256n<SEGMENT_SIZE, WARPS_PER_BLOCK, BLOCK_DIM>
          <<<gridDim, blockDim>>>(d_in_fp16, d_out, num_segments);

      PRINT_IF_ERROR(cudaEventRecord(stop));
      PRINT_IF_ERROR(cudaEventSynchronize(stop));

      // state.SkipWithError("break");
      state.PauseTiming();

      float msecTotal = 0.0f;
      PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
      state.SetIterationTime(msecTotal / 1000);
      state.ResumeTiming();
    }

    state.counters.insert({{"num_segments", num_segments},
                           {"segment_size", segment_size},
                           {"num_elements", num_segments * segment_size},
                           {"warps_per_block", WARPS_PER_BLOCK},
                           {"flops",
                            {state.iterations() * 1.0 * num_segments * segment_size,
                             benchmark::Counter::kAvgThreadsRate}}});

#if 0
  half *h_out = new half[num_elements];
  PRINT_IF_ERROR(cudaMemcpy(h_out, d_out, num_elements * sizeof(half),
                            cudaMemcpyDeviceToHost));

  int errors = 0;
  for (int j = 0; j < num_segments; j++) {
    float correct_segment_sum = 0;
    for (int i = 0; i < segment_size; i++) {
      correct_segment_sum += h_in[j * segment_size + i];
      if (fabs(half_to_float(h_out[j * segment_size + i]) -
               correct_segment_sum) > 0.1) {
        errors++;
        if (errors < 10) {
          printf("Expected %f, get h_out[%d] = %f\n", correct_segment_sum, i,
                 half_to_float(h_out[j * segment_size + i]));
        }
      }
    }
  }

  if (errors > 0) {
    printf("CUDA_UNSAFE_WMMA_SEGMENTED_PREFIXSUM_256N does not agree with SEQUENTIAL! %d "
           "errors!\n",
           errors);
  } else {
    printf("Results verified: they agree.\n\n");
  }

  delete h_out;
#endif

    cudaFree(d_in_fp16);
    cudaFree(d_out);
  } catch (...) {
    cudaFree(d_in_fp16);
    cudaFree(d_out);

    cudaDeviceReset();
    const auto p = std::current_exception();
    std::rethrow_exception(p);
  }
}

template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK>
static void iCUDA_UNSAFE_WMMA_SEGMENTED_PREFIXSUM_256N(benchmark::State &state) {
  cudaDeviceReset();
  try {
    tryCUDA_UNSAFE_WMMA_SEGMENTED_PREFIXSUM_256N<SEGMENT_SIZE, WARPS_PER_BLOCK>(state);
  } catch (const std::exception &e) {
    state.SkipWithError(e.what());
  } catch (const std::string &e) {
    state.SkipWithError(e.c_str());
  } catch (...) {
    state.SkipWithError("unknown exception");
  }
}

template <int WARPS_PER_BLOCK>
static void CUDA_UNSAFE_WMMA_SEGMENTED_PREFIXSUM_256N(benchmark::State &state) {
  const int segment_size = state.range(1);
  switch (segment_size) {
#define Dispatch(N)                                                                      \
  case N:                                                                                \
    iCUDA_UNSAFE_WMMA_SEGMENTED_PREFIXSUM_256N<N, WARPS_PER_BLOCK>(state);               \
    break

    Dispatch(256);
    Dispatch(512);
    Dispatch(1024);
    Dispatch(2048);
    Dispatch(4096);
    Dispatch(8192);
    Dispatch(16384);
    Dispatch(32768);
    Dispatch(65536);
    Dispatch(131072);
    Dispatch(262144);
    Dispatch(524288);
    Dispatch(1048576);
    Dispatch(2097152);
    Dispatch(4194304);
    Dispatch(8388608);
    Dispatch(16777216);
    Dispatch(33554432);
    Dispatch(67108864);
    Dispatch(134217728);
    Dispatch(268435456);
    Dispatch(536870912);
    Dispatch(1073741824);
    default:
      static_assert(true, "invalid segment size");
      state.SkipWithError("invalid segment size");
#undef DISPATCH
  }
}

template <int WARPS_PER_BLOCK>
static void CUDA_UNSAFE_WMMA_TUNE_SEGMENTED_PREFIXSUM_256N(benchmark::State &state) {
  CUDA_UNSAFE_WMMA_SEGMENTED_PREFIXSUM_256N<WARPS_PER_BLOCK>(state);
}
#define RUN_CUDA_UNSAFE_WMMA_TUNE(TUNE_ARGS)                                             \
  BENCHMARK_TEMPLATE(CUDA_UNSAFE_WMMA_TUNE_SEGMENTED_PREFIXSUM_256N, 1)                  \
      ->Apply(TUNE_ARGS)                                                                 \
      ->UseManualTime();                                                                 \
  BENCHMARK_TEMPLATE(CUDA_UNSAFE_WMMA_TUNE_SEGMENTED_PREFIXSUM_256N, 2)                  \
      ->Apply(TUNE_ARGS)                                                                 \
      ->UseManualTime();                                                                 \
  BENCHMARK_TEMPLATE(CUDA_UNSAFE_WMMA_TUNE_SEGMENTED_PREFIXSUM_256N, 4)                  \
      ->Apply(TUNE_ARGS)                                                                 \
      ->UseManualTime();                                                                 \
  BENCHMARK_TEMPLATE(CUDA_UNSAFE_WMMA_TUNE_SEGMENTED_PREFIXSUM_256N, 8)                  \
      ->Apply(TUNE_ARGS)                                                                 \
      ->UseManualTime();                                                                 \
  BENCHMARK_TEMPLATE(CUDA_UNSAFE_WMMA_TUNE_SEGMENTED_PREFIXSUM_256N, 16)                 \
      ->Apply(TUNE_ARGS)                                                                 \
      ->UseManualTime();

// RUN_CUDA_UNSAFE_WMMA_TUNE(Tuning256_x_14);
// RUN_CUDA_UNSAFE_WMMA_TUNE(Tuning256_x_18);
// RUN_CUDA_UNSAFE_WMMA_TUNE(Tuning256_x_22);
// RUN_CUDA_UNSAFE_WMMA_TUNE(Tuning256_x_26);
RUN_CUDA_UNSAFE_WMMA_TUNE(Tuning256_x_30);

#define RUN_CUDA_WMMA(Args)                                                              \
  BENCHMARK_TEMPLATE(CUDA_UNSAFE_WMMA_SEGMENTED_PREFIXSUM_256N, 1)                       \
      ->Args()                                                                           \
      ->UseManualTime();                                                                 \
  BENCHMARK_TEMPLATE(CUDA_UNSAFE_WMMA_SEGMENTED_PREFIXSUM_256N, 2)                       \
      ->Args()                                                                           \
      ->UseManualTime();                                                                 \
  BENCHMARK_TEMPLATE(CUDA_UNSAFE_WMMA_SEGMENTED_PREFIXSUM_256N, 4)                       \
      ->Args()                                                                           \
      ->UseManualTime();                                                                 \
  BENCHMARK_TEMPLATE(CUDA_UNSAFE_WMMA_SEGMENTED_PREFIXSUM_256N, 8)                       \
      ->Args()                                                                           \
      ->UseManualTime();                                                                 \
  BENCHMARK_TEMPLATE(CUDA_UNSAFE_WMMA_SEGMENTED_PREFIXSUM_256N, 16)                      \
      ->Args()                                                                           \
      ->UseManualTime();

RUN_CUDA_WMMA(SEG_256_ARGS);
RUN_CUDA_WMMA(SEG_512_ARGS);
RUN_CUDA_WMMA(SEG_1024_ARGS);
RUN_CUDA_WMMA(SEG_2048_ARGS);
RUN_CUDA_WMMA(SEG_4096_ARGS);
RUN_CUDA_WMMA(SEG_8192_ARGS);
RUN_CUDA_WMMA(SEG_16384_ARGS);
