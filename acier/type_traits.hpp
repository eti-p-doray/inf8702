#pragma once

#include <type_traits>
#include <utility>

namespace acier {

constexpr struct ignore_t {} ignore {};
constexpr struct blank_t {} blank {};

template <class T> using _t = typename T::type;

template <class... T>
struct always_true : std::true_type {};

template <bool C>
using when = std::enable_if_t<C, int>;
template <class... E>
using when_valid = std::enable_if_t<always_true<E...>{}, int>;

namespace callable_details {

template <class Test, class F, class... Args>
struct call_result {};
template <class F, class... Args>
struct call_result<when_valid<decltype(std::declval<F>()(std::declval<Args>()...))>,
                  F, Args...> {
  using type = decltype(std::declval<F>()(std::declval<Args>()...));
};

}

template <class F, class... Args>
struct call_result : callable_details::call_result<when<true>, F, Args...> {};

template <class F, class... Args>
using call_result_t = _t<call_result<F, Args...>>;

namespace callable_details {

template <class Test, class F, class... Args>
struct is_callable : std::false_type {};
template <class F, class... Args>
struct is_callable<when_valid<call_result_t<F, Args...>>, F, Args...>
    : std::true_type {};

template <class Test, class R, class F, class... Args>
struct is_callable_r : std::false_type {};
template <class R, class F, class... Args>
struct is_callable_r<when_valid<call_result_t<F, Args...>>, R, F, Args...>
    : std::is_convertible<call_result_t<F, Args...>, R> {};

}


template <class F, class... Args>
struct is_callable : callable_details::is_callable<when<true>, F, Args...> {};

template <class R, class F, class... Args>
struct is_callable_r : callable_details::is_callable<when<true>, R, F, Args...> {};


template <class T>
constexpr std::true_type returns(T) { return {}; }

constexpr std::true_type expect_true(std::true_type) { return {}; }
constexpr std::true_type expect_false(std::false_type) { return {}; }

}
