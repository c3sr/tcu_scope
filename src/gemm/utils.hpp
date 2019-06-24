#pragma once

#include <complex>
#include <type_traits>

#include <cuda_runtime.h>
#include <cutlass/util/util.h>

namespace gemm {
namespace detail {

  template <typename ValueT, typename AccumT>
  static const char* implementation_name();

  template <>
  const char* implementation_name<half, float>() {
    return "WGEMM";
  }
  template <>
  const char* implementation_name<__half, __half>() {
    return "HGEMM";
  }
  template <>
  const char* implementation_name<float, float>() {
    return "SGEMM";
  }
  template <>
  const char* implementation_name<double, double>() {
    return "DGEMM";
  }
  template <>
  const char* implementation_name<std::complex<float>, std::complex<float>>() {
    return "CGEMM";
  }
  template <>
  const char* implementation_name<std::complex<double>, std::complex<double>>() {
    return "ZGEMM";
  }

  template <typename T>
  static T one() {
    return T{1};
  };

  template <>
  __half one<__half>() {
    unsigned short x{1};
    __half res;
    memcpy(&res, &x, sizeof(res));
    return res;
  };

  template <typename T>
  static T zero() {
    return T{0};
  };

  template <>
  __half zero<__half>() {
    __half res;
    memset(&res, 0, sizeof(res));
    return res;
  };

  template <typename T>
  struct cuda_type {
    using type = T;
  };
  template <>
  struct cuda_type<std::complex<float>> {
    using type = cuComplex;
  };
  template <>
  struct cuda_type<std::complex<double>> {
    using type = cuDoubleComplex;
  };

} // namespace detail
} // namespace gemm
