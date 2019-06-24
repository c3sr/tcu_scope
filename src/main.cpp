
#include "config.hpp"

#include <benchmark/benchmark.h>

#include "init/flags.hpp"
#include "init/init.hpp"
#include "utils/utils.hpp"

extern void doCUDA_WMMA_GEMM_ACCURACY(int M, int N, int K);
extern void doCUDA_WMMA_GEMV_ACCURACY(int M, int N);

// const char *bench_git_refspec();
// const char *bench_git_hash();
// const char *bench_git_tag();

// std::string bench_get_build_version() {
//   return fmt::format("{} {} {}", bench_git_refspec(), bench_git_tag(),
//   bench_git_hash());
// }

static int run_benchmark(int argc, char **argv) {
  try {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv))
      return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
  } catch (const std::exception &e) {
    LOG(error, "ERROR:: Exception = {}", e.what());
    return 1;
  } catch (const std::string &e) {
    LOG(error, "ERROR:: Exception = {}", e);
    return 1;
  } catch (...) {
    const auto p = std::current_exception();
    LOG(error, "ERROR:: Uknown Exception");
    return 1;
  }
}

int main(int argc, char *argv[]) {

  std::vector<char *> argList(argv, argv + argc);
  argList.push_back((char *) "--benchmark_counters_tabular=true");

  argv = argList.data();
  argc = argList.size();

  init(argc, argv);

  // cudaProfilerStart();

  return run_benchmark(argc, argv);

  // cudaProfilerStop();

  /* doCUDA_WMMA_GEMM_ACCURACY(64, 64, 64); */
  /* doCUDA_WMMA_GEMV_ACCURACY(64, 64); */
}
