#define CUB_HALF_OPTIMIZATION 1

#include <benchmark/benchmark.h>

#include <type_traits>
#include <utility>

#include <cooperative_groups.h>

#include "init/init.hpp"
#include "unsafe_reduction/args.hpp"
#include "utils/utils.hpp"

#include "kernel.cuh"

using namespace wmma_unsafe_reduction;

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

template <size_t SEGMENT_SIZE, int WARPS_PER_BLOCK>
void tryCUDA_UNSAFE_WMMA_FULL_REDUCTION_CG(benchmark::State &state) {
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
  PRINT_IF_ERROR(cudaMalloc(&d_out, gridDim.x * sizeof(half)));
  PRINT_IF_ERROR(cudaMemset(d_out, 0, gridDim.x * sizeof(half)));

  cuda_memory_set(d_in_fp16, 0.001f, num_elements);

  cudaEvent_t start, stop;
  PRINT_IF_ERROR(cudaEventCreate(&start));
  PRINT_IF_ERROR(cudaEventCreate(&stop));

  defer(cudaEventDestroy(start));
  defer(cudaEventDestroy(stop));

#if 0
  const auto params =
      collect_arguments(d_in_fp16, d_out, num_segments, SEGMENT_SIZE);

  defer(free(params));
#else
  void *params[] = {(void *) &d_in_fp16, (void *) &d_out, (void *) &num_segments};
#endif

  int maxActiveBlocks;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, (void *) &compute_wmma_reduction_cg<WARPS_PER_BLOCK, BLOCK_DIM>,
      blockDim.x, 0);
  // printf("gridDim = %d maxActiveBlocks = %d\n",gridDim.x,
  // maxActiveBlocks);

  try {
    for (auto _ : state) {
      PRINT_IF_ERROR(cudaMemset(d_out, 0, gridDim.x * sizeof(half)));
      PRINT_IF_ERROR(cudaEventRecord(start));

      cudaLaunchCooperativeKernel(
          (const void
               *) &compute_wmma_reduction_cg<SEGMENT_SIZE, WARPS_PER_BLOCK, BLOCK_DIM>,
          gridDim, blockDim, params);

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
    printf(
        "CUDA_UNSAFE_WMMA_FULL_REDUCTION_CG does not agree with SEQUENTIAL! %d errors!\n",
        errors);
  } else {
    printf("Results verified: they agree.\n\n");
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
template <int SEGMENT_SIZE, int WARPS_PER_BLOCK>
void CUDA_UNSAFE_WMMA_FULL_REDUCTION_CG(benchmark::State &state) {
  cudaDeviceReset();
  try {
    tryCUDA_UNSAFE_WMMA_FULL_REDUCTION_CG<SEGMENT_SIZE, WARPS_PER_BLOCK>(state);
  } catch (const std::exception &e) {
    state.SkipWithError(e.what());
  } catch (const std::string &e) {
    state.SkipWithError(e.c_str());
  } catch (...) {
    state.SkipWithError("unknown exception");
  }
}

#define BENCHMARK_REDUCTION0(SEGMENT_SIZE, WARPS_PER_BLOCK)                              \
  BENCHMARK_TEMPLATE(CUDA_UNSAFE_WMMA_FULL_REDUCTION_CG, SEGMENT_SIZE, WARPS_PER_BLOCK)  \
      ->ARGS()                                                                           \
      ->UseManualTime()

#define BENCHMARK_REDUCTION(SEGMENT_SIZE)                                                \
  BENCHMARK_REDUCTION0(SEGMENT_SIZE, 1);                                                 \
  BENCHMARK_REDUCTION0(SEGMENT_SIZE, 2);                                                 \
  BENCHMARK_REDUCTION0(SEGMENT_SIZE, 4);                                                 \
  BENCHMARK_REDUCTION0(SEGMENT_SIZE, 8);                                                 \
  BENCHMARK_REDUCTION0(SEGMENT_SIZE, 16)

#if 0 // disabled
BENCHMARK_REDUCTION(256);
BENCHMARK_REDUCTION(2 * 256);
BENCHMARK_REDUCTION(4 * 256);
BENCHMARK_REDUCTION(8 * 256);
#if 0 // uses too much shared mem
BENCHMARK_REDUCTION(16 * 256);
BENCHMARK_REDUCTION(32 * 256);
BENCHMARK_REDUCTION(64 * 256);
BENCHMARK_REDUCTION(128 * 256);
BENCHMARK_REDUCTION(256 * 256);
BENCHMARK_REDUCTION(512 * 256);
BENCHMARK_REDUCTION(1024 * 256);
#endif
#endif
