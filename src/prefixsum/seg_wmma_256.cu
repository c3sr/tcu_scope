
#include <benchmark/benchmark.h>

#include "init/init.hpp"
#include "prefixsum/args.hpp"
#include "utils/utils.hpp"

#include "kernel.cuh"

using namespace wmma_prefixsum;

template <int SEGMENTS_PER_WARP, int WARPS_PER_BLOCK>
void tryCUDA_WMMA_SEGMENTED_PREFIXSUM_256(benchmark::State &state) {
  const size_t num_segments = state.range(0);
  const size_t segment_size = state.range(1);

  if (segment_size != WMMA_TILE_SIZE) {
    state.SkipWithError("segment size must be WMMA_TILE_SIZE (256)");
  }

  const int BLOCK_DIM          = WARPS_PER_BLOCK * WARP_SIZE;
  const size_t num_elements    = num_segments * segment_size;
  const int segments_per_block = WARPS_PER_BLOCK * SEGMENTS_PER_WARP;

  half *d_in_fp16;
  half *d_out;

  try {
    PRINT_IF_ERROR(cudaMalloc(&d_in_fp16, num_elements * sizeof(half)));
    PRINT_IF_ERROR(cudaMalloc(&d_out, num_elements * sizeof(half)));

    cuda_memory_set(d_in_fp16, 0.001f, num_elements);

    dim3 gridDim, blockDim;
    blockDim.x = BLOCK_DIM;
    gridDim.x  = (num_segments + segments_per_block - 1) / segments_per_block;

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

      compute_wmma_segmented_prefixsum_256<SEGMENTS_PER_WARP, WARPS_PER_BLOCK, BLOCK_DIM>
          <<<gridDim, blockDim>>>(d_in_fp16, d_out, num_segments);

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
                           {"segments_per_warp", SEGMENTS_PER_WARP},
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
               correct_segment_sum) > 0.01) {
        errors++;
        if (errors < 10) {
          printf("Expected %f, get h_out[%d] = %f\n", correct_segment_sum, i,
               half_to_float(h_out[j * segment_size + i]));
        }
      }
    }
  }

  if (errors > 0) {
    printf("CUDA_WMMA_SEGMENTED_PREFIXSUM_256 does not agree with SEQUENTIAL! "
           "%d errors!\n",
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

template <int SEGMENTS_PER_WARP, int WARPS_PER_BLOCK>
void CUDA_WMMA_SEGMENTED_PREFIXSUM_256(benchmark::State &state) {
  cudaDeviceReset();
  try {
    tryCUDA_WMMA_SEGMENTED_PREFIXSUM_256<SEGMENTS_PER_WARP, WARPS_PER_BLOCK>(state);
  } catch (const std::exception &e) {
    state.SkipWithError(e.what());
  } catch (const std::string &e) {
    state.SkipWithError(e.c_str());
  } catch (...) {
    state.SkipWithError("unknown exception");
  }
}

#define RUN_CUDA_WMMA0(SEGMENTS_PER_WARP, WARPS_PER_BLOCK)                               \
  BENCHMARK_TEMPLATE(                                                                    \
      CUDA_WMMA_SEGMENTED_PREFIXSUM_256, SEGMENTS_PER_WARP, WARPS_PER_BLOCK)             \
      ->SEG_256_ARGS()                                                                   \
      ->UseManualTime();

#define RUN_CUDA_WMMA(SEGMENTS_PER_WARP)                                                 \
  RUN_CUDA_WMMA0(SEGMENTS_PER_WARP, 1);                                                  \
  RUN_CUDA_WMMA0(SEGMENTS_PER_WARP, 2);                                                  \
  RUN_CUDA_WMMA0(SEGMENTS_PER_WARP, 4);                                                  \
  RUN_CUDA_WMMA0(SEGMENTS_PER_WARP, 8);

RUN_CUDA_WMMA(1);
RUN_CUDA_WMMA(2);
RUN_CUDA_WMMA(4);
RUN_CUDA_WMMA(8);
