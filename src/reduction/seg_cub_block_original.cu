
#include <benchmark/benchmark.h>

#include "init/init.hpp"
#include "reduction/args.hpp"
#include "utils/utils.hpp"

#include <cub/cub.cuh>

using namespace cub;

template <int THREADS_PER_BLOCK, int ITEMS_PER_THREAD, BlockReduceAlgorithm ALGORITHM>
__global__ void compute_cub_block_segmented_reduction(half *d_in, half *d_out) {
  // Specialize BlockReduce type for our thread block
  typedef BlockReduce<half, THREADS_PER_BLOCK, ALGORITHM> BlockReduceT;
  // Shared memory
  __shared__ typename BlockReduceT::TempStorage temp_storage;
  // Per-thread tile data
  half data[ITEMS_PER_THREAD];
  LoadDirectStriped<THREADS_PER_BLOCK, half, ITEMS_PER_THREAD>(
      threadIdx.x, d_in + blockIdx.x * THREADS_PER_BLOCK * ITEMS_PER_THREAD, data);
  // Compute sum
  half aggregate = BlockReduceT(temp_storage).Sum(data);
  // Store aggregate
  if (threadIdx.x == 0) {
    d_out[blockIdx.x] = aggregate;
  }
}

template <int THREADS_PER_BLOCK, int ITEMS_PER_THREAD, BlockReduceAlgorithm ALGORITHM>
static void ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION(benchmark::State &state) {
  const size_t num_segments = state.range(0);
  const size_t segment_size = state.range(1);

  if (segment_size != THREADS_PER_BLOCK * ITEMS_PER_THREAD) {
    state.SkipWithError("segment size must be THREADS_PER_BLOCK x ITEMS_PER_THREAD");
    return;
  }

  if (num_segments >= CUDA_MAX_GRID_SIZE) {
    state.SkipWithError(
        fmt::format("gridDim.x={} is greater than CUDA_MAX_GRID_SIZE", num_segments)
            .c_str());
    return;
  }

  const size_t num_elements = num_segments * segment_size;

  half *d_in_fp16 = nullptr;
  half *d_out     = nullptr;

  try {
    PRINT_IF_ERROR(cudaMalloc(&d_in_fp16, num_elements * sizeof(half)));
    PRINT_IF_ERROR(cudaMalloc(&d_out, num_segments * sizeof(half)));

    cuda_memory_set(d_in_fp16, 0.001f, num_elements);

    PRINT_IF_ERROR(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    PRINT_IF_ERROR(cudaEventCreate(&start));
    PRINT_IF_ERROR(cudaEventCreate(&stop));

    defer(cudaEventDestroy(start));
    defer(cudaEventDestroy(stop));

    for (auto _ : state) {
      PRINT_IF_ERROR(cudaEventRecord(start));

      compute_cub_block_segmented_reduction<THREADS_PER_BLOCK,
                                            ITEMS_PER_THREAD,
                                            ALGORITHM>
          <<<num_segments, THREADS_PER_BLOCK>>>(d_in_fp16, d_out);

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
                           {"threads_per_block", THREADS_PER_BLOCK},
                           {"items_per_thread", ITEMS_PER_THREAD},
                           {"block_reduce_algorithm", (int) ALGORITHM},
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
    printf("ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION does not agree with SEQUENTIAL! %d "
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

// BlockReduceAlgorithm are BLOCK_REDUCE_RAKING,
// BLOCK_REDUCE_WARP_REDUCTIONS and
// BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY
// BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY
#if 1
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 32, 1, BLOCK_REDUCE_RAKING)
    ->SEG_32_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 32, 2, BLOCK_REDUCE_RAKING)
    ->SEG_64_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 64, 1, BLOCK_REDUCE_RAKING)
    ->SEG_64_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 32, 4, BLOCK_REDUCE_RAKING)
    ->SEG_128_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 64, 2, BLOCK_REDUCE_RAKING)
    ->SEG_128_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 128, 1, BLOCK_REDUCE_RAKING)
    ->SEG_128_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 32, 8, BLOCK_REDUCE_RAKING)
    ->SEG_256_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 64, 4, BLOCK_REDUCE_RAKING)
    ->SEG_256_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 128, 2, BLOCK_REDUCE_RAKING)
    ->SEG_256_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 256, 1, BLOCK_REDUCE_RAKING)
    ->SEG_256_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 32, 16, BLOCK_REDUCE_RAKING)
    ->SEG_512_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 64, 8, BLOCK_REDUCE_RAKING)
    ->SEG_512_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 128, 4, BLOCK_REDUCE_RAKING)
    ->SEG_512_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 256, 2, BLOCK_REDUCE_RAKING)
    ->SEG_512_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 512, 1, BLOCK_REDUCE_RAKING)
    ->SEG_512_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 32, 32, BLOCK_REDUCE_RAKING)
    ->SEG_1024_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 64, 16, BLOCK_REDUCE_RAKING)
    ->SEG_1024_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 128, 8, BLOCK_REDUCE_RAKING)
    ->SEG_1024_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 256, 4, BLOCK_REDUCE_RAKING)
    ->SEG_1024_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 512, 2, BLOCK_REDUCE_RAKING)
    ->SEG_1024_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 1024, 1, BLOCK_REDUCE_RAKING)
    ->SEG_1024_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 32, 64, BLOCK_REDUCE_RAKING)
    ->SEG_2048_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 64, 32, BLOCK_REDUCE_RAKING)
    ->SEG_2048_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 128, 16, BLOCK_REDUCE_RAKING)
    ->SEG_2048_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 256, 8, BLOCK_REDUCE_RAKING)
    ->SEG_2048_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 512, 4, BLOCK_REDUCE_RAKING)
    ->SEG_2048_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 1024, 2, BLOCK_REDUCE_RAKING)
    ->SEG_2048_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 32, 128, BLOCK_REDUCE_RAKING)
    ->SEG_4096_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 64, 64, BLOCK_REDUCE_RAKING)
    ->SEG_4096_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 128, 32, BLOCK_REDUCE_RAKING)
    ->SEG_4096_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 256, 16, BLOCK_REDUCE_RAKING)
    ->SEG_4096_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 512, 8, BLOCK_REDUCE_RAKING)
    ->SEG_4096_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 1024, 4, BLOCK_REDUCE_RAKING)
    ->SEG_4096_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 32, 256, BLOCK_REDUCE_RAKING)
    ->SEG_8192_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 64, 128, BLOCK_REDUCE_RAKING)
    ->SEG_8192_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 128, 64, BLOCK_REDUCE_RAKING)
    ->SEG_8192_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 256, 32, BLOCK_REDUCE_RAKING)
    ->SEG_8192_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 512, 16, BLOCK_REDUCE_RAKING)
    ->SEG_8192_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 1024, 8, BLOCK_REDUCE_RAKING)
    ->SEG_8192_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 32, 512, BLOCK_REDUCE_RAKING)
    ->SEG_16384_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 64, 256, BLOCK_REDUCE_RAKING)
    ->SEG_16384_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 128, 128, BLOCK_REDUCE_RAKING)
    ->SEG_16384_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 256, 64, BLOCK_REDUCE_RAKING)
    ->SEG_16384_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 512, 32, BLOCK_REDUCE_RAKING)
    ->SEG_16384_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_REDUCTION, 1024, 16, BLOCK_REDUCE_RAKING)
    ->SEG_16384_ARGS()
    ->UseManualTime();
#endif
