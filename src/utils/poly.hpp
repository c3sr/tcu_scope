#pragma once
#if 0
#include <cuda_fp16.h>

namespace detail {

template <class... Tn>
struct reverse;

template <__half... coefficients>
struct horner;
template <__half coefficient>
struct horner {
  static __half operator()(const __half arg) {
    return arg * coefficient;
  }
};
template <__half... head, __half last>
struct horner {
  static __half operator()(const __half arg) {
    return last * arg + horner<head...>::operator()(arg);
  }
};

}; // namespace detail

template <__half... args>
struct polynomial {
  static constexpr std::initializer_list<__half> coefficients = detail::reverse(std::forward<__half>(args)...);
  static constexpr int order                                  = coefficients.size();
  static __half eval_horner(const __half arg) {
    using f = detail::horner<coefficients>;
    return f(arg);
  }
};
#endif
