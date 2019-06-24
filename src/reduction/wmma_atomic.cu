#define CUB_HALF_OPTIMIZATION 1

#include <benchmark/benchmark.h>

#include <type_traits>
#include <utility>

#include "init/init.hpp"
#include "reduction/args.hpp"
#include "utils/utils.hpp"

#include "kernel.cuh"

using namespace wmma_reduction;

enum class block_synchronization_stategy : int { synchronize_threads, atomic_ballot };

template <size_t SEGMENT_SIZE,
          int WARPS_PER_BLOCK,
          block_synchronization_stategy sync_stategy>
void tryCUDA_WMMA_FULL_REDUCTION_ATOMIC(benchmark::State &state) {
  const size_t num_elements = state.range(0);

  if (num_elements % SEGMENT_SIZE) {
    state.SkipWithError("num_elements must be multiples of SEGMENT_SIZE");
    return;
  }

  size_t num_segments = (num_elements + SEGMENT_SIZE - 1) / SEGMENT_SIZE;
  const int BLOCK_DIM = WARPS_PER_BLOCK * WARP_SIZE;

  half *d_in_fp16 = nullptr;
  half *d_out     = nullptr;

  dim3 gridDim, blockDim;
  blockDim.x = BLOCK_DIM;
  gridDim.x  = (num_segments + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

  if (gridDim.x >= CUDA_MAX_GRID_SIZE) {
    state.SkipWithError(
        fmt::format("gridDim.x={} is greater than CUDA_MAX_GRID_SIZE", gridDim.x)
            .c_str());
    return;
  }

  PRINT_IF_ERROR(cudaMalloc(&d_in_fp16, num_elements * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc(&d_out, 2 * sizeof(half)));

  PRINT_IF_ERROR(cudaMemset(d_out, 0, 2 * sizeof(half)));

  cuda_memory_set(d_in_fp16, 0.001f, num_elements);

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  defer(cudaEventDestroy(start));
  defer(cudaEventDestroy(stop));

  try {
    for (auto _ : state) {
      PRINT_IF_ERROR(cudaMemset(d_out, 0, 2 * sizeof(half)));
      PRINT_IF_ERROR(cudaEventRecord(start));

      if (sync_stategy == block_synchronization_stategy::synchronize_threads) {
        compute_wmma_reduction_atomic_w_syncthreads<SEGMENT_SIZE,
                                                    WARPS_PER_BLOCK,
                                                    BLOCK_DIM>
            <<<gridDim, blockDim>>>(d_in_fp16, d_out, num_segments);
      } else if (sync_stategy == block_synchronization_stategy::atomic_ballot) {
        compute_wmma_reduction_atomic_w_atomicballot<SEGMENT_SIZE,
                                                     WARPS_PER_BLOCK,
                                                     BLOCK_DIM>
            <<<gridDim, blockDim>>>(d_in_fp16, d_out, num_segments);
      }
      PRINT_IF_ERROR(cudaEventRecord(stop));
      PRINT_IF_ERROR(cudaEventSynchronize(stop));

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
  half h_out;
  PRINT_IF_ERROR(
      cudaMemcpy(&h_out, d_out,  sizeof(half), cudaMemcpyDeviceToHost));

  int errors        = 0;
  float correct_sum = 0;
  for (int i = 0; i < num_elements; i++) {
    correct_sum += h_in[i];
  }
  if (fabs(half_to_float(h_out) - correct_sum) > 0.1) {
    errors++;
    if (errors < 10) {
    printf("Expected Reuction = %f, got h_out_buf = %f\n", correct_sum,
           half_to_float(h_out));
    }
  }

  if (errors > 0) {
    printf(
        "CUDA_WMMA_FULL_REDUCTION_ATOMIC does not agree with SEQUENTIAL! %d errors!\n",
        errors);
  }
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
template <size_t SEGMENT_SIZE,
          int WARPS_PER_BLOCK,
          block_synchronization_stategy sync_stategy>
void CUDA_WMMA_FULL_REDUCTION_ATOMIC(benchmark::State &state) {
  cudaDeviceReset();
  try {
    tryCUDA_WMMA_FULL_REDUCTION_ATOMIC<SEGMENT_SIZE, WARPS_PER_BLOCK, sync_stategy>(
        state);
  } catch (const std::exception &e) {
    state.SkipWithError(e.what());
  } catch (const std::string &e) {
    state.SkipWithError(e.c_str());
  } catch (...) {
    state.SkipWithError("unknown exception");
  }
}

template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK>
void CUDA_WMMA_FULL_REDUCTION_ATOMIC_W_BLOCK_SYNC(benchmark::State &state) {
  CUDA_WMMA_FULL_REDUCTION_ATOMIC<SEGMENT_SIZE,
                                  WARPS_PER_BLOCK,
                                  block_synchronization_stategy::synchronize_threads>(
      state);
}

template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK>
void CUDA_WMMA_FULL_REDUCTION_ATOMIC_W_ATOMIC_BALLOT(benchmark::State &state) {
  CUDA_WMMA_FULL_REDUCTION_ATOMIC<SEGMENT_SIZE,
                                  WARPS_PER_BLOCK,
                                  block_synchronization_stategy::atomic_ballot>(state);
}

#define BENCHMARK_REDUCTION0(SEGMENT_SIZE, WARPS_PER_BLOCK)                              \
  BENCHMARK_TEMPLATE(                                                                    \
      CUDA_WMMA_FULL_REDUCTION_ATOMIC_W_ATOMIC_BALLOT, SEGMENT_SIZE, WARPS_PER_BLOCK)    \
      ->ARGS()                                                                           \
      ->UseManualTime();                                                                 \
  BENCHMARK_TEMPLATE(                                                                    \
      CUDA_WMMA_FULL_REDUCTION_ATOMIC_W_BLOCK_SYNC, SEGMENT_SIZE, WARPS_PER_BLOCK)       \
      ->ARGS()                                                                           \
      ->UseManualTime()

#define BENCHMARK_REDUCTION(SEGMENT_SIZE)                                                \
  BENCHMARK_REDUCTION0(SEGMENT_SIZE, 1);                                                 \
  BENCHMARK_REDUCTION0(SEGMENT_SIZE, 2);                                                 \
  BENCHMARK_REDUCTION0(SEGMENT_SIZE, 4);                                                 \
  BENCHMARK_REDUCTION0(SEGMENT_SIZE, 8);                                                 \
  BENCHMARK_REDUCTION0(SEGMENT_SIZE, 16)

BENCHMARK_REDUCTION(256);
BENCHMARK_REDUCTION(2 * 256);
BENCHMARK_REDUCTION(4 * 256);
BENCHMARK_REDUCTION(8 * 256);
BENCHMARK_REDUCTION(16 * 256);
// BENCHMARK_REDUCTION(32 * 256);
// BENCHMARK_REDUCTION(64 * 256);
// BENCHMARK_REDUCTION(128 * 256);
// BENCHMARK_REDUCTION(256 * 256);
// BENCHMARK_REDUCTION(512 * 256);
// BENCHMARK_REDUCTION(1024 * 256);
