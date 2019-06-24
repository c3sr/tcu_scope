#define CUB_HALF_OPTIMIZATION 1

#include <benchmark/benchmark.h>

#include <type_traits>
#include <utility>

#include "init/init.hpp"
#include "reduction/args.hpp"
#include "utils/utils.hpp"

#include "kernel.cuh"
#include <cub/cub.cuh>

using namespace wmma_reduction;

template <size_t BASE_SEGMENT_SIZE, size_t SEGMENT_SIZE, int WARPS_PER_BLOCK>
void tryCUDA_WMMA_FULL_REDUCTION_2KERS_BLOCK(benchmark::State &state) {
  const int BLOCK_DIM       = WARPS_PER_BLOCK * WARP_SIZE;
  const size_t num_elements = state.range(0);
  const size_t num_segments = (num_elements + SEGMENT_SIZE - 1) / SEGMENT_SIZE;

  if (num_elements % SEGMENT_SIZE) {
    state.SkipWithError("num_elements must be multiples of SEGMENT_SIZE");
    return;
  }

  half *d_in_fp16  = nullptr;
  half *d_out      = nullptr;
  half *d_temp_out = nullptr;

  PRINT_IF_ERROR(cudaMalloc(&d_in_fp16, num_elements * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc(&d_out, 1 * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc(&d_temp_out, num_segments * sizeof(half)));

  cuda_memory_set(d_in_fp16, 0.001f, num_elements);

  dim3 gridDim, blockDim;
  blockDim.x = BLOCK_DIM;
  gridDim.x  = num_segments;

  if (gridDim.x >= CUDA_MAX_GRID_SIZE) {
    state.SkipWithError(
        fmt::format("gridDim.x={} is greater than CUDA_MAX_GRID_SIZE", gridDim.x)
            .c_str());
    return;
  }

  void *d_temp_storage      = NULL;
  size_t temp_storage_bytes = 0;
  PRINT_IF_ERROR(cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_temp_out,
                                        d_out, num_segments));
  PRINT_IF_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  defer(cudaEventDestroy(start));
  defer(cudaEventDestroy(stop));

  try {
    for (auto _ : state) {
      PRINT_IF_ERROR(cudaEventRecord(start));
      switch (BASE_SEGMENT_SIZE) {
        case 16:
          compute_wmma_segmented_reduction_16n_block<SEGMENT_SIZE, WARPS_PER_BLOCK,
                                                     BLOCK_DIM>
              <<<gridDim, blockDim>>>(d_in_fp16, d_temp_out, num_segments);
          break;
        case 256:
          compute_wmma_segmented_reduction_256n_block<half, SEGMENT_SIZE, WARPS_PER_BLOCK,
                                                      BLOCK_DIM>
              <<<gridDim, blockDim>>>(d_in_fp16, d_temp_out, num_segments);
          break;
        default:
          static_assert(true, "only 16 and 256 base segment sizes are support");
      }

#if 1
      cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_temp_out, d_out,
                             num_segments);
#else
      compute_wmma_warp_reduction<1, BLOCK_DIM>
          <<<1, WARP_SIZE>>>(d_temp_out, d_out, 1, num_segments);
#endif

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
        cudaMemcpy(&h_out, d_out, 1 * sizeof(half), cudaMemcpyDeviceToHost));

    int errors        = 0;
    float correct_sum = 0;
    for (int i = 0; i < num_elements; i++) {
      correct_sum += h_in[i];
    }
    if (fabs(half_to_float(h_out) - correct_sum) > 0.001) {
      errors++;
      printf("Expected Reuction = %f, got h_out = %f\n", correct_sum,
             half_to_float(h_out));
    }

    if (errors > 0) {
      printf("CUDA_WMMA_FULL_REDUCTION does not agree with SEQUENTIAL! %d "
             "errors!\n",
             errors);
    } else {
      printf("Results verified: they agree.\n\n");
    }
#endif

    cudaFree(d_in_fp16);
    cudaFree(d_out);
    cudaFree(d_temp_out);
    cudaFree(d_temp_storage);
  } catch (...) {
    cudaFree(d_in_fp16);
    cudaFree(d_out);
    cudaFree(d_temp_out);
    cudaFree(d_temp_storage);

    cudaDeviceReset();
    const auto p = std::current_exception();
    std::rethrow_exception(p);
  }
}

template <size_t BASE_SEGMENT_SIZE, size_t SEGMENT_SIZE, int WARPS_PER_BLOCK>
void CUDA_WMMA_FULL_REDUCTION_2KERS_BLOCK(benchmark::State &state) {
  cudaDeviceReset();
  try {
    tryCUDA_WMMA_FULL_REDUCTION_2KERS_BLOCK<BASE_SEGMENT_SIZE, SEGMENT_SIZE,
                                            WARPS_PER_BLOCK>(state);
  } catch (const std::exception &e) {
    state.SkipWithError(e.what());
  } catch (const std::string &e) {
    state.SkipWithError(e.c_str());
  } catch (...) {
    state.SkipWithError("unknown exception");
  }
}

template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK>
void CUDA_WMMA_FULL_REDUCTION_2KERS_BLOCK_BASE_16(benchmark::State &state) {
  CUDA_WMMA_FULL_REDUCTION_2KERS_BLOCK<16, SEGMENT_SIZE, WARPS_PER_BLOCK>(state);
}

template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK>
void CUDA_WMMA_FULL_REDUCTION_2KERS_BLOCK_BASE_256(benchmark::State &state) {
  CUDA_WMMA_FULL_REDUCTION_2KERS_BLOCK<256, SEGMENT_SIZE, WARPS_PER_BLOCK>(state);
}

#define BENCHMARK_REDUCTION0(SEGMENT_SIZE, WARPS_PER_BLOCK)                              \
  BENCHMARK_TEMPLATE(CUDA_WMMA_FULL_REDUCTION_2KERS_BLOCK_BASE_16, SEGMENT_SIZE,         \
                     WARPS_PER_BLOCK)                                                    \
      ->ARGS()                                                                           \
      ->UseManualTime();                                                                 \
  BENCHMARK_TEMPLATE(CUDA_WMMA_FULL_REDUCTION_2KERS_BLOCK_BASE_256, SEGMENT_SIZE,        \
                     WARPS_PER_BLOCK)                                                    \
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
BENCHMARK_REDUCTION(32 * 256);
BENCHMARK_REDUCTION(64 * 256);
BENCHMARK_REDUCTION(128 * 256);
BENCHMARK_REDUCTION(256 * 256);
/* BENCHMARK_REDUCTION(512 * 256); */
/* BENCHMARK_REDUCTION(1024 * 256); */
