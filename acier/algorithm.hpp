#pragma once

#include <algorithm>

template <class T>
T saturate(T val, T min, T max) {
  return std::min(std::max(val, min), max);
}

template <class T, class U>
T saturate(const U& val) {
  return saturate(val, U(std::numeric_limits<T>::min()), U(std::numeric_limits<T>::max()));
}
