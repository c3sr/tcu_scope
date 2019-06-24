#pragma once

#include "benchmark/benchmark.h"

#if 0

class Benchmark : public benchmark::internal::Benchmark {
public:
  typedef std::map<std::string, std::string> MetaDataType;
  Benchmark* AddMetaData(const std::string& key, const std::string& val) {
    meta_data_[key] = val;
    return this;
  }
  const MetaDataType& MetaData() const {
    return meta_data_;
  }

  BENCHMARK_ALWAYS_INLINE
  void SetFlops(size_t flops) {
    flops_ = flops;
  }

  BENCHMARK_ALWAYS_INLINE
  size_t flops() const {
    return flops_;
  }

private:
  MetaDataType meta_data_;
  size_t flops_{0};
};



static bool setBenchmarkFormat(const char* format) {
    if (!strcmp(format, "tabular")) {
        gBenchmarkReporter.reset(new benchmark::ConsoleReporter());
    } else if (!strcmp(format, "json")) {
        gBenchmarkReporter.reset(new benchmark::JSONReporter());
    } else if (!strcmp(format, "csv")) {
        gBenchmarkReporter.reset(new benchmark::CSVReporter());
    } else {
        fprintf(stderr, "Unknown format '%s'", format);
        return false;
    }
    return true;
}


void BenchmarkReporter::PrintBasicContext(std::ostream *out,
                                          Context const &context) {
  CHECK(out) << "cannot be null";
  auto &Out = *out;

  Out << LocalDateTimeString() << "\n";

  const CPUInfo &info = context.cpu_info;
  Out << "Run on (" << info.num_cpus << " X "
      << (info.cycles_per_second / 1000000.0) << " MHz CPU "
      << ((info.num_cpus > 1) ? "s" : "") << ")\n";
  if (info.caches.size() != 0) {
    Out << "CPU Caches:\n";
    for (auto &CInfo : info.caches) {
      Out << "  L" << CInfo.level << " " << CInfo.type << " "
          << (CInfo.size / 1000) << "K";
      if (CInfo.num_sharing != 0)
        Out << " (x" << (info.num_cpus / CInfo.num_sharing) << ")";
      Out << "\n";
    }
  }

  if (info.scaling_enabled) {
    Out << "***WARNING*** CPU scaling is enabled, the benchmark "
           "real time measurements may be noisy and will incur extra "
           "overhead.\n";
  }

#ifndef NDEBUG
  Out << "***WARNING*** Library was built as DEBUG. Timings may be "
         "affected.\n";
#endif
}

class BenchmarkReporter : public JSONReporter {
public:
  JSONReporter() : first_report_(true) {
  }
  virtual bool ReportContext(const Context& context);
  virtual void ReportRuns(const std::vector<Run>& reports);
  virtual void Finalize();

private:
  void PrintRunData(const Run& report);

  bool first_report_;
};

#endif