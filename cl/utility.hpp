#pragma once

#include "acier/algorithm.hpp"

namespace cl {

template <class T, class... Args>
constexpr T flags(Args... args) {
  return acier::accumulate(std::initializer_list<T>{static_cast<T>(args)...}, T(), acier::bit_or());
}

}
