#define CUB_HALF_OPTIMIZATION 1

#include <benchmark/benchmark.h>

#include "init/init.hpp"
#include "prefixsum/args.hpp"
#include "utils/utils.hpp"

#include "kernel.cuh"
#include <cub/cub.cuh>

using namespace wmma_unsafe_prefixsum;

template <int SEGMENT_SIZE, int WARPS_PER_BLOCK>
void tryCUDA_UNSAFE_WMMA_FULL_PREFIXSUM_3KERS(benchmark::State &state) {
  const int BLOCK_DIM       = WARPS_PER_BLOCK * WARP_SIZE;
  const size_t num_elements = state.range(0);
  const size_t num_segments = (num_elements + SEGMENT_SIZE - 1) / SEGMENT_SIZE;

  if (num_elements % SEGMENT_SIZE) {
    state.SkipWithError("num_elements must be multiples of SEGMENT_SIZE");
    return;
  }

  half *d_in_fp16    = nullptr;
  half *d_out        = nullptr;
  half *partial_sums = nullptr;

  PRINT_IF_ERROR(cudaMalloc(&d_in_fp16, num_elements * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc(&d_out, 1 * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc(&partial_sums, num_segments * sizeof(half)));

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

  void *d_temp_storage      = NULL;
  size_t temp_storage_bytes = 0;
  PRINT_IF_ERROR(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                               partial_sums, partial_sums, num_segments));
  PRINT_IF_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  defer(cudaEventDestroy(start));
  defer(cudaEventDestroy(stop));

  try {
    for (auto _ : state) {
      PRINT_IF_ERROR(cudaEventRecord(start));

      compute_wmma_segmented_prefixsum_256n_ps<SEGMENT_SIZE, WARPS_PER_BLOCK, BLOCK_DIM>
          <<<gridDim, blockDim>>>(d_in_fp16, d_out, partial_sums, num_segments);

      cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, partial_sums,
                                    partial_sums, num_segments);

      add_partial_sums<256, SEGMENT_SIZE>
          <<<num_segments, 256>>>(d_out, partial_sums, num_elements);

      PRINT_IF_ERROR(cudaEventRecord(stop));
      PRINT_IF_ERROR(cudaEventSynchronize(stop));

      // state.SkipWithError("break");
      state.PauseTiming();

      float msecTotal = 0.0f;
      PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
      state.SetIterationTime(msecTotal / 1000);
      state.ResumeTiming();
    }

    state.counters.insert({{"num_elements", num_elements},
                           {"num_segments", num_segments},
                           {"segment_size", SEGMENT_SIZE},
                           {"warps_per_block", WARPS_PER_BLOCK},
                           {"flops",
                            {state.iterations() * 1.0 * num_elements,
                             benchmark::Counter::kAvgThreadsRate}}});

#if 0
    half *h_out = new half[num_elements];
    PRINT_IF_ERROR(cudaMemcpy(h_out, d_out, num_elements * sizeof(half),
                              cudaMemcpyDeviceToHost));

    int errors        = 0;
    float correct_sum = 0;

    for (int i = 0; i < num_elements; i++) {
      correct_sum += h_in[i];
      if (fabs(half_to_float(h_out[i]) - correct_sum) > 0.1) {
        errors++;
        if (errors < 10) {
          printf("Expected %f, get h_out[%d] = %f\n", correct_sum, i,
                 half_to_float(h_out[i]));
        }
      }
    }

    if (errors > 0) {
      printf("CUDA_PREFIXSUM_WMM does not agree with SEQUENTIAL! %d errors!\n",
             errors);
    } else {
      printf("Results verified: they agree.\n\n");
    }
#if 0
    half h_partial_sums[num_segments];
    PRINT_IF_ERROR(cudaMemcpy(h_partial_sums, partial_sums,
                              sizeof(half) * num_segments,
                              cudaMemcpyDeviceToHost));
    for (int i = 0; i < num_segments; i++) {
      printf("-------partial_sums[%d] = %f\n", i,
             half_to_float(h_partial_sums[i]));
    }

    delete h_out;
#endif
#endif
    cudaFree(d_in_fp16);
    cudaFree(d_out);
    cudaFree(partial_sums);
    cudaFree(d_temp_storage);

  } catch (...) {
    cudaFree(d_in_fp16);
    cudaFree(d_out);
    cudaFree(partial_sums);
    cudaFree(d_temp_storage);

    cudaDeviceReset();
    const auto p = std::current_exception();
    std::rethrow_exception(p);
  }
}

template <int SEGMENT_SIZE, int WARPS_PER_BLOCK>
void CUDA_UNSAFE_WMMA_FULL_PREFIXSUM_3KERS(benchmark::State &state) {
  cudaDeviceReset();
  try {
    tryCUDA_UNSAFE_WMMA_FULL_PREFIXSUM_3KERS<SEGMENT_SIZE, WARPS_PER_BLOCK>(state);
  } catch (const std::exception &e) {
    state.SkipWithError(e.what());
  } catch (const std::string &e) {
    state.SkipWithError(e.c_str());
  } catch (...) {
    state.SkipWithError("unknown exception");
  }
}

#define BENCHMARK_PRIFIXSUM0(SEGMENT_SIZE, WARPS_PER_BLOCK)                              \
  BENCHMARK_TEMPLATE(CUDA_UNSAFE_WMMA_FULL_PREFIXSUM_3KERS, SEGMENT_SIZE,                \
                     WARPS_PER_BLOCK)                                                    \
      ->ARGS()                                                                           \
      ->UseManualTime()

#define BENCHMARK_PRIFIXSUM(SEGMENT_SIZE)                                                \
  BENCHMARK_PRIFIXSUM0(SEGMENT_SIZE, 1);                                                 \
  BENCHMARK_PRIFIXSUM0(SEGMENT_SIZE, 2);                                                 \
  BENCHMARK_PRIFIXSUM0(SEGMENT_SIZE, 4);                                                 \
  BENCHMARK_PRIFIXSUM0(SEGMENT_SIZE, 8);                                                 \
  BENCHMARK_PRIFIXSUM0(SEGMENT_SIZE, 16)

BENCHMARK_PRIFIXSUM(256);
BENCHMARK_PRIFIXSUM(2 * 256);
BENCHMARK_PRIFIXSUM(4 * 256);
BENCHMARK_PRIFIXSUM(8 * 256);
BENCHMARK_PRIFIXSUM(16 * 256);
/* BENCHMARK_PRIFIXSUM(32 * 256); */
/* BENCHMARK_PRIFIXSUM(64 * 256); */
/* BENCHMARK_PRIFIXSUM(128 * 256); */
/* BENCHMARK_PRIFIXSUM(256 * 256); */
/* BENCHMARK_PRIFIXSUM(512 * 256); */
/* BENCHMARK_PRIFIXSUM(1024 * 256); */
