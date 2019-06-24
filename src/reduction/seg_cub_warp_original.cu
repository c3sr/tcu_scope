
#include <benchmark/benchmark.h>

#include "init/init.hpp"
#include "reduction/args.hpp"
#include "utils/utils.hpp"

#include <cub/cub.cuh>

using namespace cub;

#ifndef WARP_SIZE
#define WARP_SIZE (32)
#endif // WARP_SIZE

template <int THREADS_PER_BLOCK, int LOGICAL_THREADS_PER_WARP>
__global__ void compute_cub_warp_segmented_reduction(half *d_in, half *d_out) {
  constexpr int num_warps = THREADS_PER_BLOCK / LOGICAL_THREADS_PER_WARP;
  const int offset        = blockIdx.x * blockDim.x + threadIdx.x;
  const int warp_id       = threadIdx.x / LOGICAL_THREADS_PER_WARP;
  const int lane_id       = threadIdx.x % LOGICAL_THREADS_PER_WARP;

  typedef WarpReduce<half, LOGICAL_THREADS_PER_WARP> WarpReduceT;
  __shared__ typename WarpReduceT::TempStorage temp_storage[num_warps];

  half thread_data = d_in[offset];
  WarpReduceT(temp_storage[warp_id]).Sum(thread_data);
  if (lane_id == 0 ) {
    d_out[offset/LOGICAL_THREADS_PER_WARP] = thread_data;
  }
}

template <int THREADS_PER_BLOCK, int LOGICAL_THREADS_PER_WARP>
static void ORIGINAL_CUB_WARP_SEGMENTED_REDUCTION(benchmark::State &state) {
  const size_t num_segments = state.range(0);
  const size_t segment_size = state.range(1);

  if (segment_size != LOGICAL_THREADS_PER_WARP) {
    state.SkipWithError("segment size must be LOGICAL_THREADS_PER_WARP");
  }

  const size_t num_elements    = num_segments * segment_size;
  const int segments_per_block = THREADS_PER_BLOCK / LOGICAL_THREADS_PER_WARP;

  dim3 gridDim, blockDim;
  blockDim.x = THREADS_PER_BLOCK;
  gridDim.x  = (num_segments + segments_per_block - 1) / segments_per_block;

  if (gridDim.x >= CUDA_MAX_GRID_SIZE) {
    state.SkipWithError(
        fmt::format("gridDim.x={} is greater than CUDA_MAX_GRID_SIZE", gridDim.x)
            .c_str());
    return;
  }

  half *d_in_fp16 = nullptr;
  half *d_out     = nullptr;
  cudaEvent_t start, stop;

  defer(cudaDeviceReset());

  try {
    PRINT_IF_ERROR(cudaMalloc(&d_in_fp16, num_elements * sizeof(half)));
    PRINT_IF_ERROR(cudaMalloc(&d_out, num_segments * sizeof(half)));

    cuda_memory_set(d_in_fp16, 0.001f, num_elements);

    PRINT_IF_ERROR(cudaDeviceSynchronize());

    PRINT_IF_ERROR(cudaEventCreate(&start));
    PRINT_IF_ERROR(cudaEventCreate(&stop));

    defer(cudaEventDestroy(start));
    defer(cudaEventDestroy(stop));

    for (auto _ : state) {
      PRINT_IF_ERROR(cudaEventRecord(start));

      compute_cub_warp_segmented_reduction<THREADS_PER_BLOCK, LOGICAL_THREADS_PER_WARP>
          <<<gridDim, blockDim>>>(d_in_fp16, d_out);

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
                           {"logical_threads_per_block", LOGICAL_THREADS_PER_WARP},
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
               correct_segment_sum) > 0.001) {
        errors++;
        printf("Expected %f, get h_out[%d] = %f\n", correct_segment_sum, i,
               half_to_float(h_out[j * segment_size + i]));
      }
    }
  }

  if (errors > 0) {
    printf("CUB_SEGMENTED_REDUCTION_16 does not agree with SEQUENTIAL! %d "
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

BENCHMARK_TEMPLATE(ORIGINAL_CUB_WARP_SEGMENTED_REDUCTION, 32, 16)->SEG_16_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_WARP_SEGMENTED_REDUCTION, 64, 16)->SEG_16_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_WARP_SEGMENTED_REDUCTION, 128, 16)->SEG_16_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_WARP_SEGMENTED_REDUCTION, 256, 16)->SEG_16_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_WARP_SEGMENTED_REDUCTION, 512, 16)->SEG_16_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_WARP_SEGMENTED_REDUCTION, 1024, 16)
    ->SEG_16_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_WARP_SEGMENTED_REDUCTION, 32, 32)->SEG_32_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_WARP_SEGMENTED_REDUCTION, 64, 32)->SEG_32_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_WARP_SEGMENTED_REDUCTION, 128, 32)->SEG_32_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_WARP_SEGMENTED_REDUCTION, 256, 32)->SEG_32_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_WARP_SEGMENTED_REDUCTION, 512, 32)->SEG_32_ARGS()->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_WARP_SEGMENTED_REDUCTION, 1024, 32)
    ->SEG_32_ARGS()
    ->UseManualTime();
