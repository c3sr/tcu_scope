#define CUB_HALF_OPTIMIZATION 1

#include <benchmark/benchmark.h>

#include "init/init.hpp"
#include "unsafe_prefixsum/args.hpp"
#include "utils/utils.hpp"

#include "kernel.cuh"
#include <cub/cub.cuh>

using namespace wmma_unsafe_prefixsum;

template <int SEGMENTS_PER_WARP, int WARPS_PER_BLOCK>
void CUDA_UNSAFE_WMMA_FULL_PREFIXSUM_3KERS_256(benchmark::State &state) {
  const int num_elements = state.range(0);

  if (num_elements % WMMA_TILE_SIZE) {
    state.SkipWithError("num_elements must be multiples of WMMA_TILE_SIZE");
  }

  float *d_in_fp32;
  half *d_in_fp16, *d_out, *partial_sums;

  float *h_in = new float[num_elements];
  for (int i = 0; i < num_elements; i++) {
    h_in[i] = 0.01f;
  }

  const int BLOCK_DIM          = WARPS_PER_BLOCK * WARP_SIZE;
  const int num_segments       = (num_elements + WMMA_TILE_SIZE - 1) / WMMA_TILE_SIZE;
  const int segments_per_block = WARPS_PER_BLOCK * SEGMENTS_PER_WARP;

  PRINT_IF_ERROR(cudaMalloc(&d_in_fp32, num_elements * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc(&d_in_fp16, num_elements * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc(&d_out, num_elements * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc((void **) &partial_sums, num_segments * sizeof(half)));

  PRINT_IF_ERROR(
      cudaMemcpy(d_in_fp32, h_in, num_elements * sizeof(float), cudaMemcpyHostToDevice));

  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(num_elements + 1023) / 1024, 1024>>>(
      d_in_fp16, d_in_fp32, num_elements)));

  dim3 gridDim, blockDim;
  blockDim.x = BLOCK_DIM;
  gridDim.x  = (num_segments + segments_per_block - 1) / segments_per_block;

  void *d_temp_storage      = NULL;
  size_t temp_storage_bytes = 0;
  PRINT_IF_ERROR(cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes,
                                               partial_sums, partial_sums, num_segments));
  PRINT_IF_ERROR(cudaMalloc(&d_temp_storage, temp_storage_bytes));

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  for (auto _ : state) {
    PRINT_IF_ERROR(cudaEventRecord(start));

    compute_wmma_segmented_prefixsum_256_ps<SEGMENTS_PER_WARP, WARPS_PER_BLOCK, BLOCK_DIM>
        <<<gridDim, blockDim>>>(d_in_fp16, d_out, partial_sums, num_segments);

    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, partial_sums,
                                  partial_sums, num_segments);

    add_partial_sums<256, WMMA_TILE_SIZE>
        <<<num_segments, 256>>>(d_out, partial_sums, num_elements);

    PRINT_IF_ERROR(cudaEventRecord(stop));
    PRINT_IF_ERROR(cudaEventSynchronize(stop));

    /* state.SkipWithError("break"); */
    state.PauseTiming();

    float msecTotal = 0.0f;
    PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
    state.SetIterationTime(msecTotal / 1000);
    state.ResumeTiming();
  }

  state.counters.insert(
      {{"num_elements", num_elements},
       {"segments_per_warp", SEGMENTS_PER_WARP},
       {"warps_per_block", WARPS_PER_BLOCK},
       {"flops",
        {state.iterations() * 1.0 * num_elements, benchmark::Counter::kAvgThreadsRate}}});

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

  delete h_out;
#endif

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  PRINT_IF_ERROR(cudaFree(d_in_fp16));
  PRINT_IF_ERROR(cudaFree(d_in_fp32));
  PRINT_IF_ERROR(cudaFree(d_out));
  PRINT_IF_ERROR(cudaFree(partial_sums));

  cudaDeviceReset();

  delete h_in;
}

#if 0
BENCHMARK_TEMPLATE(CUDA_UNSAFE_WMMA_FULL_PREFIXSUM_3KERS_256, 2, 4)
    ->ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(CUDA_UNSAFE_WMMA_FULL_PREFIXSUM_3KERS_256, 2, 8)
    ->ARGS()
    ->UseManualTime();
#endif
