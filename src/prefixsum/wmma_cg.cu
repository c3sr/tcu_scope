#if 0
#define CUB_HALF_OPTIMIZATION 1

#include <benchmark/benchmark.h>

#include <type_traits>
#include <utility>

#include <cooperative_groups.h>

#include "init/init.hpp"
#include "prefixsum/args.hpp"
#include "utils/utils.hpp"

#include "kernel.cuh"

using namespace wmma_prefixsum;

template <typename Fun>
struct is_function_ptr
    : std::integral_constant<
          bool, std::is_pointer<Fun>::value and
                    std::is_function<typename std::remove_pointer<Fun>::type>::value> {};

template <typename Arg, typename... Args>
static inline void collect_argument_addresses(void **collected_addresses, Arg &&arg,
                                              Args &&... args) {
  collected_addresses[0] = static_cast<void *>(&arg);
  collect_argument_addresses(collected_addresses + 1, std::forward<Args>(args)...);
}

template <typename... Args>
static inline void **collect_arguments(Args &&... args) {
  void **argument_ptrs = (void **) malloc((sizeof...(Args)) * sizeof(void *));
  collect_argument_addresses(argument_ptrs, std::forward<Args>(args)...);
  return argument_ptrs;
}

template <int SEGMENT_SIZE, int WARPS_PER_BLOCK>
void tryCUDA_WMMA_FULL_PREFIXSUM_CG(benchmark::State &state) {
  const int num_elements = state.range(0);

  int num_segments    = (num_elements + SEGMENT_SIZE - 1) / SEGMENT_SIZE;
  const int BLOCK_DIM = WARPS_PER_BLOCK * WARP_SIZE;

  if (num_elements % SEGMENT_SIZE) {
    state.SkipWithError("num_elements must be multiples of SEGMENT_SIZE");
    return;
  }

  auto h_in = new float[num_elements];
  std::fill(h_in, h_in + num_elements, 0.01f);

  float *d_in_fp32   = nullptr;
  half *d_in_fp16    = nullptr;
  half *d_out        = nullptr;
  half *partial_sums = nullptr;

  PRINT_IF_ERROR(cudaMalloc(&d_in_fp32, num_elements * sizeof(float)));
  PRINT_IF_ERROR(cudaMalloc(&d_in_fp16, num_elements * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc(&d_out, num_elements * sizeof(half)));
  PRINT_IF_ERROR(cudaMalloc(&partial_sums, num_segments * sizeof(half)));

  PRINT_IF_ERROR(
      cudaMemcpy(d_in_fp32, h_in, num_elements * sizeof(float), cudaMemcpyHostToDevice));

  PRINT_IF_LAUNCH_ERROR((convertFp32ToFp16<<<(num_elements + 1023) / 1024, 1024>>>(
      d_in_fp16, d_in_fp32, num_elements)));

  dim3 gridDim, blockDim;
  blockDim.x = BLOCK_DIM;
  gridDim.x  = (num_segments + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

#if 0
  const auto params =
      collect_arguments(d_in_fp16, d_out, num_segments, SEGMENT_SIZE);

  defer(free(params));
#else
  const auto segment_size = SEGMENT_SIZE;
  void *params[]          = {(void *) &d_in_fp16, (void *) &d_out, (void *) &partial_sums,
                    (void *) &num_segments, (void *) &segment_size};
#endif

  int maxActiveBlocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks,
      (void *) &compute_wmma_prefixsum_cg<SEGMENT_SIZE, WARPS_PER_BLOCK, BLOCK_DIM>,
      blockDim.x, 0);
  /* printf("gridDim = %d maxActiveBlocks = %d\n", gridDim.x,
   * maxActiveBlocks);
   */

  try {
    for (auto _ : state) {
      PRINT_IF_ERROR(cudaEventRecord(start));

      cudaLaunchCooperativeKernel(
          (const void
               *) &compute_wmma_prefixsum_cg<SEGMENT_SIZE, WARPS_PER_BLOCK, BLOCK_DIM>,
          gridDim, blockDim, params);

      PRINT_IF_ERROR(cudaEventRecord(stop));
      PRINT_IF_ERROR(cudaEventSynchronize(stop));

      /* state.SkipWithError("break"); */
      state.PauseTiming();

      float msecTotal = 0.0f;
      PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
      state.SetIterationTime(msecTotal / 1000);
      state.ResumeTiming();
    }

    state.counters.insert({{"num_elements", num_elements},
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

    delete h_out;
#endif

    PRINT_IF_ERROR(cudaEventDestroy(start));
    PRINT_IF_ERROR(cudaEventDestroy(stop));
  } catch (...) {
    delete[] h_in;
    cudaFree(d_in_fp32);
    cudaFree(d_in_fp16);
    cudaFree(d_out);
    cudaFree(partial_sums);
    const auto p = std::current_exception();
    std::rethrow_exception(p);
  }
}

template <int SEGMENT_SIZE, int WARPS_PER_BLOCK>
void CUDA_WMMA_FULL_PREFIXSUM_CG(benchmark::State &state) {
  cudaDeviceReset();
  try {
    tryCUDA_WMMA_FULL_PREFIXSUM_CG<SEGMENT_SIZE, WARPS_PER_BLOCK>(state);
  } catch (const std::exception &e) {
    state.SkipWithError(e.what());
  } catch (const std::string &e) {
    state.SkipWithError(e.c_str());
  } catch (...) {
    state.SkipWithError("unknown exception");
  }
}

#define BENCHMARK_PRIFIXSUM0(SEGMENT_SIZE, WARPS_PER_BLOCK)                              \
  BENCHMARK_TEMPLATE(CUDA_WMMA_FULL_PREFIXSUM_CG, SEGMENT_SIZE, WARPS_PER_BLOCK)         \
      ->ARGS()                                                                           \
      ->UseManualTime()

#define BENCHMARK_PRIFIXSUM(SEGMENT_SIZE)                                                \
  BENCHMARK_PRIFIXSUM0(SEGMENT_SIZE, 1);                                                 \
  BENCHMARK_PRIFIXSUM0(SEGMENT_SIZE, 2);                                                 \
  BENCHMARK_PRIFIXSUM0(SEGMENT_SIZE, 4);                                                 \
  BENCHMARK_PRIFIXSUM0(SEGMENT_SIZE, 8);                                                 \
  BENCHMARK_PRIFIXSUM0(SEGMENT_SIZE, 16)

// disabled
#if 0
BENCHMARK_PRIFIXSUM(256);
BENCHMARK_PRIFIXSUM(2 * 256);
BENCHMARK_PRIFIXSUM(4 * 256);
BENCHMARK_PRIFIXSUM(8 * 256);
#if 0 // use too much shared data
BENCHMARK_PRIFIXSUM(16 * 256);
BENCHMARK_PRIFIXSUM(32 * 256);
BENCHMARK_PRIFIXSUM(64 * 256);
BENCHMARK_PRIFIXSUM(128 * 256);
BENCHMARK_PRIFIXSUM(256 * 256);
BENCHMARK_PRIFIXSUM(512 * 256);
BENCHMARK_PRIFIXSUM(1024 * 256);
#endif
#endif

#endif
