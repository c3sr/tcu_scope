
#include <benchmark/benchmark.h>

#include "init/init.hpp"
#include "reduction/args.hpp"
#include "utils/utils.hpp"

#include <thrust/execution_policy.h>
#include <thrust/reduce.h>

static void THRUST_FULL_REDUCTION(benchmark::State &state) {
  const size_t num_elements = state.range(0);

  cudaEvent_t start, stop;
  half *d_in_fp16 = nullptr;
  half h_out; // thrust::reduce return a quantity that must be
              // deposited in a
              // host resident variable

  try {
    PRINT_IF_ERROR(cudaMalloc(&d_in_fp16, num_elements * sizeof(half)));

    cuda_memory_set(d_in_fp16, 0.001f, num_elements);

    PRINT_IF_ERROR(cudaDeviceSynchronize());

    PRINT_IF_ERROR(cudaEventCreate(&start));
    PRINT_IF_ERROR(cudaEventCreate(&stop));

    defer(cudaEventDestroy(start));
    defer(cudaEventDestroy(stop));

    for (auto _ : state) {
      PRINT_IF_ERROR(cudaEventRecord(start));

      h_out = thrust::reduce(thrust::device, d_in_fp16, d_in_fp16 + num_elements);

      PRINT_IF_ERROR(cudaEventRecord(stop));
      PRINT_IF_ERROR(cudaEventSynchronize(stop));

      state.PauseTiming();

      float msecTotal = 0.0f;
      PRINT_IF_ERROR(cudaEventElapsedTime(&msecTotal, start, stop));
      state.SetIterationTime(msecTotal / 1000);
      state.ResumeTiming();
    }

    (void) h_out;

    state.counters.insert({{"num_elements", num_elements},
                           {"flops",
                            {state.iterations() * 1.0 * num_elements,
                             benchmark::Counter::kAvgThreadsRate}}});

#if 0
  int errors        = 0;
  float correct_sum = 0;
  for (int i = 0; i < num_elements; i++) {
    correct_sum += h_in[i];
  }

  if (fabs(half_to_float(h_out) - correct_sum) > 0.001) {
    errors++;
    if (errors < 10) {
      printf("Expected %f, get h_out = %f\n", correct_sum,
             half_to_float(h_out));
    }
  }

  if (errors > 0) {
    printf("THRUST_FULL_REDUCTION does not agree with SEQUENTIAL! %d errors!\n",
           errors);
  } else {
    printf("Results verified: they agree.\n\n");
  }
#endif

    cudaFree(d_in_fp16);
  } catch (...) {
    cudaFree(d_in_fp16);

    cudaDeviceReset();
    const auto p = std::current_exception();
    std::rethrow_exception(p);
  }
}

BENCHMARK(THRUST_FULL_REDUCTION)->ARGS()->UseManualTime();
