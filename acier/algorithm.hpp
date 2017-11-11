#pragma once

#include <algorithm>
#include <iterator>

#include "acier/operators.hpp"

namespace acier {

template <class T>
T saturate(T val, T min, T max) {
  return std::min(std::max(val, min), max);
}

template <class T, class U>
T saturate(const U& val) {
  return saturate(val, U(std::numeric_limits<T>::min()), U(std::numeric_limits<T>::max()));
}

template <class Rng, class F = identity>
constexpr bool all_of(Rng&& rng, F&& fn = {}) {
  auto last = std::end(rng);
  for (auto it = std::begin(rng); it != last; ++it) {
    if (!fn(*it))
      return false;
  }
  return true;
}

template <class Rng, class F = identity>
constexpr bool any_of(Rng&& rng, F&& fn = {}) {
  auto last = std::end(rng);
  for (auto it = std::begin(rng); it != last; ++it) {
    if (fn(*it))
      return true;
  }
  return false;
}

template <class Rng, class T, class BinaryOperation = plus>
constexpr T accumulate(Rng&& rng, T init, BinaryOperation op) {
  auto last = std::end(rng);
  for (auto it = std::begin(rng); it != last; ++it) {
    init = op(init, *it);
  }
  return init;
}

}
