
#include <benchmark/benchmark.h>

#include "init/init.hpp"
#include "prefixsum/args.hpp"
#include "utils/utils.hpp"

#include <cub/cub.cuh>

using namespace cub;

template <int THREADS_PER_BLOCK, int ITEMS_PER_THREAD, BlockScanAlgorithm ALGORITHM>
__global__ void compute_cub_block_segmented_prefixsum(half *d_in, half *d_out) {
  // Specialize BlockLoad type for our thread block (uses warp-striped
  // loads for
  // coalescing, then transposes in shared memory to a blocked
  // arrangement)
  typedef BlockLoad<half, THREADS_PER_BLOCK, ITEMS_PER_THREAD, BLOCK_LOAD_WARP_TRANSPOSE>
      BlockLoadT;
  // Specialize BlockStore type for our thread block (uses
  // warp-striped loads
  // for coalescing, then transposes in shared memory to a blocked
  // arrangement)
  typedef BlockStore<half,
                     THREADS_PER_BLOCK,
                     ITEMS_PER_THREAD,
                     BLOCK_STORE_WARP_TRANSPOSE>
      BlockStoreT;
  // Specialize BlockScan type for our thread block
  typedef BlockScan<half, THREADS_PER_BLOCK, ALGORITHM> BlockScanT;
  // Shared memory
  __shared__ union TempStorage {
    typename BlockLoadT::TempStorage load;
    typename BlockStoreT::TempStorage store;
    typename BlockScanT::TempStorage scan;
  } temp_storage;
  // Per-thread tile data
  half data[ITEMS_PER_THREAD];
  int offset = blockIdx.x * THREADS_PER_BLOCK * ITEMS_PER_THREAD;
  // Load items into a blocked arrangement
  BlockLoadT(temp_storage.load).Load(d_in + offset, data);
  // Barrier for smem reuse
  __syncthreads();
  // Compute inclusive prefix sum
  BlockScanT(temp_storage.scan).InclusiveSum(data, data);
  // Barrier for smem reuse
  __syncthreads();
  // Store items from a blocked arrangement
  BlockStoreT(temp_storage.store).Store(d_out + offset, data);
}

template <int THREADS_PER_BLOCK, int ITEMS_PER_THREAD, BlockScanAlgorithm ALGORITHM>
static void ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM(benchmark::State &state) {
  const size_t num_segments = state.range(0);
  const size_t segment_size = state.range(1);
  const size_t num_elements = num_segments * segment_size;

  if (segment_size != THREADS_PER_BLOCK * ITEMS_PER_THREAD) {
    state.SkipWithError("segment size must be THREADS_PER_BLOCK x ITEMS_PER_THREAD");
  }

  if (num_segments >= CUDA_MAX_GRID_SIZE) {
    state.SkipWithError(
        fmt::format("gridDim.x={} is greater than CUDA_MAX_GRID_SIZE", num_segments)
            .c_str());
    return;
  }

  half *d_in_fp16 = nullptr;
  half *d_out     = nullptr;

  try {
    PRINT_IF_ERROR(cudaMalloc(&d_in_fp16, num_elements * sizeof(half)));
    PRINT_IF_ERROR(cudaMalloc(&d_out, num_elements * sizeof(half)));

    cuda_memory_set(d_in_fp16, 0.001f, num_elements);

    PRINT_IF_ERROR(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    PRINT_IF_ERROR(cudaEventCreate(&start));
    PRINT_IF_ERROR(cudaEventCreate(&stop));

    defer(cudaEventDestroy(start));
    defer(cudaEventDestroy(stop));

    for (auto _ : state) {
      PRINT_IF_ERROR(cudaEventRecord(start));

      compute_cub_block_segmented_prefixsum<THREADS_PER_BLOCK,
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

    const double giga_bandwidth =
        state.iterations() * num_segments * segment_size / 1000.0 / 1000.0;

    state.counters.insert(
        {{"num_segments", num_segments},
         {"segment_size", segment_size},
         {"num_elements", num_segments * segment_size},
         {"giga_bandwidth", giga_bandwidth},
         {"threads_per_block", THREADS_PER_BLOCK},
         {"items_per_thread", ITEMS_PER_THREAD},
         {"block_scan_algorithm", (int) ALGORITHM},
         {"percent_peek_bandwidth", giga_bandwidth / device_giga_bandwidth * 100.0},
         {"flops",
          {state.iterations() * 1.0 * num_segments * segment_size,
           benchmark::Counter::kAvgThreadsRate}}});

#if 0
  half *h_out = new half[num_elements];
  PRINT_IF_ERROR(cudaMemcpy(h_out, d_out, num_elements * sizeof(half), cudaMemcpyDeviceToHost));

  int errors = 0;
  for (int j = 0; j < num_segments; j++) {
    float correct_segment_sum = 0;
    for (int i = 0; i < segment_size; i++) {
      correct_segment_sum += h_in[j * segment_size + i];
      if (fabs(half_to_float(h_out[j * segment_size + i]) - correct_segment_sum) > 0.001) {
        errors++;
        printf("Expected %f, get h_out[%d] = %f\n", correct_segment_sum, i, half_to_float(h_out[j * segment_size + i]));
      }
    }
  }

  if (errors > 0) {
    printf("ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM does not agree with SEQUENTIAL! %d errors!\n", errors);
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

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 32, 1, BLOCK_SCAN_RAKING)
    ->SEG_32_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 32, 2, BLOCK_SCAN_RAKING)
    ->SEG_64_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 64, 1, BLOCK_SCAN_RAKING)
    ->SEG_64_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 32, 4, BLOCK_SCAN_RAKING)
    ->SEG_128_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 64, 2, BLOCK_SCAN_RAKING)
    ->SEG_128_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 128, 1, BLOCK_SCAN_RAKING)
    ->SEG_128_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 32, 8, BLOCK_SCAN_RAKING)
    ->SEG_256_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 64, 4, BLOCK_SCAN_RAKING)
    ->SEG_256_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 128, 2, BLOCK_SCAN_RAKING)
    ->SEG_256_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 256, 1, BLOCK_SCAN_RAKING)
    ->SEG_256_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 32, 16, BLOCK_SCAN_RAKING)
    ->SEG_512_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 64, 8, BLOCK_SCAN_RAKING)
    ->SEG_512_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 128, 4, BLOCK_SCAN_RAKING)
    ->SEG_512_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 256, 2, BLOCK_SCAN_RAKING)
    ->SEG_512_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 512, 1, BLOCK_SCAN_RAKING)
    ->SEG_512_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 32, 32, BLOCK_SCAN_RAKING)
    ->SEG_1024_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 64, 16, BLOCK_SCAN_RAKING)
    ->SEG_1024_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 128, 8, BLOCK_SCAN_RAKING)
    ->SEG_1024_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 256, 4, BLOCK_SCAN_RAKING)
    ->SEG_1024_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 512, 2, BLOCK_SCAN_RAKING)
    ->SEG_1024_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 1024, 1, BLOCK_SCAN_RAKING)
    ->SEG_1024_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 32, 64, BLOCK_SCAN_RAKING)
    ->SEG_2048_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 64, 32, BLOCK_SCAN_RAKING)
    ->SEG_2048_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 128, 16, BLOCK_SCAN_RAKING)
    ->SEG_2048_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 256, 8, BLOCK_SCAN_RAKING)
    ->SEG_2048_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 512, 4, BLOCK_SCAN_RAKING)
    ->SEG_2048_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 1024, 2, BLOCK_SCAN_RAKING)
    ->SEG_2048_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 32, 128, BLOCK_SCAN_RAKING)
    ->SEG_4096_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 64, 64, BLOCK_SCAN_RAKING)
    ->SEG_4096_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 128, 32, BLOCK_SCAN_RAKING)
    ->SEG_4096_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 256, 16, BLOCK_SCAN_RAKING)
    ->SEG_4096_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 512, 8, BLOCK_SCAN_RAKING)
    ->SEG_4096_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 1024, 4, BLOCK_SCAN_RAKING)
    ->SEG_4096_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 32, 256, BLOCK_SCAN_RAKING)
    ->SEG_8192_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 64, 128, BLOCK_SCAN_RAKING)
    ->SEG_8192_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 128, 64, BLOCK_SCAN_RAKING)
    ->SEG_8192_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 256, 32, BLOCK_SCAN_RAKING)
    ->SEG_8192_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 512, 16, BLOCK_SCAN_RAKING)
    ->SEG_8192_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 1024, 8, BLOCK_SCAN_RAKING)
    ->SEG_8192_ARGS()
    ->UseManualTime();

BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 32, 512, BLOCK_SCAN_RAKING)
    ->SEG_16384_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 64, 256, BLOCK_SCAN_RAKING)
    ->SEG_16384_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 128, 128, BLOCK_SCAN_RAKING)
    ->SEG_16384_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 256, 64, BLOCK_SCAN_RAKING)
    ->SEG_16384_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 512, 32, BLOCK_SCAN_RAKING)
    ->SEG_16384_ARGS()
    ->UseManualTime();
BENCHMARK_TEMPLATE(ORIGINAL_CUB_BLOCK_SEGMENTED_PREFIXSUM, 1024, 16, BLOCK_SCAN_RAKING)
    ->SEG_16384_ARGS()
    ->UseManualTime();
