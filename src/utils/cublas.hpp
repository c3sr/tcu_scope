#pragma once

#include <cublas_v2.h>

#include "utils/error.hpp"

namespace utils {
namespace detail {

  template <>
  ALWAYS_INLINE const char *error_string<cublasStatus_t>(const cublasStatus_t &status) {
    switch (status) {
      case CUBLAS_STATUS_SUCCESS:
        return "CUBLAS_STATUS_SUCCESS";
      case CUBLAS_STATUS_NOT_INITIALIZED:
        return "CUBLAS_STATUS_NOT_INITIALIZED";
      case CUBLAS_STATUS_ALLOC_FAILED:
        return "CUBLAS_STATUS_ALLOC_FAILED";
      case CUBLAS_STATUS_INVALID_VALUE:
        return "CUBLAS_STATUS_INVALID_VALUE";
      case CUBLAS_STATUS_ARCH_MISMATCH:
        return "CUBLAS_STATUS_ARCH_MISMATCH";
      case CUBLAS_STATUS_MAPPING_ERROR:
        return "CUBLAS_STATUS_MAPPING_ERROR";
      case CUBLAS_STATUS_EXECUTION_FAILED:
        return "CUBLAS_STATUS_EXECUTION_FAILED";
      case CUBLAS_STATUS_INTERNAL_ERROR:
        return "CUBLAS_STATUS_INTERNAL_ERROR";
      case CUBLAS_STATUS_NOT_SUPPORTED:
        return "CUBLAS_STATUS_NOT_SUPPORTED";
      default:
        return "Unknown CUBLAS error.";
    }
  }

  template <>
  ALWAYS_INLINE bool is_success<cublasStatus_t>(const cublasStatus_t &err) {
    return err == CUBLAS_STATUS_SUCCESS;
  }

} // namespace detail
} // namespace utils