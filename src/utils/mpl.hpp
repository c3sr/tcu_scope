#pragma once

#include <type_traits>

namespace mpl {
template <size_t... Ns>
struct mul;

template <>
struct mul<> {
  using value_type                  = size_t;
  static constexpr value_type value = 1;
};

template <size_t N, size_t... Ns>
struct mul<N, Ns...> {
  using value_type                  = size_t;
  static constexpr value_type value = N * mul<Ns...>::value;
};

template <size_t... Ns>
struct add;

template <>
struct add<> {
  using value_type                  = size_t;
  static constexpr value_type value = 0;
};

template <size_t N, size_t... Ns>
struct add<N, Ns...> {
  using value_type                  = size_t;
  static constexpr value_type value = N + add<Ns...>::value;
};

template <typename T, typename U, typename... Ts>
struct is_all_same
    : std::integral_constant<bool,
                             is_all_same<T, U>::value & is_all_same<U, Ts...>::value> {};

template <typename T>
struct is_all_same<T, T> : std::true_type {};

template <typename T, typename U>
struct is_all_same<T, U> : std::false_type {};

template <bool cond, class T = void>
using enable_if_t = typename std::enable_if<cond, T>::type;

template <typename... xs>
struct list;

template <typename x, typename xs>
struct cons;

template <typename x, typename... xs>
struct cons<x, list<xs...>> {
  using type = list<x, xs...>;
};

template <typename xs>
struct rest;

template <typename x, typename... xs>
struct rest<list<x, xs...>> {
  using type = list<xs...>;
};
} // namespace mpl