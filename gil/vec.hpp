#pragma once

#include <array>
#include <algorithm>

#include "acier/algorithm.hpp"
#include "acier/operators.hpp"
#include "acier/type_traits.hpp"

namespace gil {

template <class T, size_t N>
class vec : public std::array<T, N> {
 public:
  using std::array<T, N>::array;
  vec() = default;
  vec(std::initializer_list<T> init) {
    std::copy(init.begin(), init.end(), this->begin());
  }
  vec(const std::array<T, N>& that) : std::array<T, N>(that) {}
  template <class U>
  vec(const std::array<U, N>& that) {
    std::copy(that.begin(), that.end(), this->begin());
  }
};

template <class T, size_t N, class F>
vec<acier::call_result_t<F, T>, N> apply(const vec<T, N>& a, const F& fcn) {
  vec<acier::call_result_t<F, T>, N> x;
  auto a_it = a.begin();
  auto x_it = x.begin();
  for (; a_it != a.end(); ++a_it, ++x_it) {
    *x_it = fcn(*a_it);
  }
  return x;
}

template <class T, class U, size_t N, class F>
vec<acier::call_result_t<F, T, U>, N> apply(const vec<T, N>& a, const vec<U, N>& b, const F& fcn) {
  vec<acier::call_result_t<F, T, U>, N> x;
  auto a_it = a.begin();
  auto b_it = b.begin();
  auto x_it = x.begin();
  for (; a_it != a.end(); ++a_it, ++b_it, ++x_it) {
    *x_it = fcn(*a_it, *b_it);
  }
  return x;
}

template <class T, size_t N, class F>
void for_each(vec<T, N>& a, const F& fcn) {
  auto a_it = a.begin();
  for (; a_it != a.end(); ++a_it) {
    fcn(*a_it);
  }
}

template <class T, class U, size_t N, class F>
void for_each(vec<T, N>& a, const vec<U, N>& b, const F& fcn) {
  auto a_it = a.begin();
  auto b_it = b.begin();
  for (; a_it != a.end(); ++a_it, ++b_it) {
    fcn(*a_it, *b_it);
  }
}

template <class T, size_t N>
T norm2(const vec<T, N>& x) {
  T n = {};
  for (auto v : x) {
    n += v*v;
  }
  return n;
}

template <class T, class U, size_t N>
bool operator<(const vec<T, N>& a, const vec<U, N>& b) {
  return norm2(a) < norm2(b);
}

template <class T, class U, size_t N>
bool operator>(const vec<T, N>& a, const vec<U, N>& b) {
  return !(b < a);
}

template <class T, class U, size_t N>
vec<std::common_type_t<T, U>, N> operator+(const vec<T, N>& a, const vec<U, N>& b) {
  return apply(a, b, acier::plus());
}

template <class T, class U, size_t N, acier::when<std::is_scalar<U>::value> = true>
vec<std::common_type_t<T, U>, N> operator+(const vec<T, N>& a, const U& b) {
  using namespace std::placeholders;
  return apply(a, std::bind(acier::plus(), _1, b));
}

template <class T, class U, size_t N>
vec<std::common_type_t<T, U>, N> operator-(const vec<T, N>& a, const vec<U, N>& b) {
  return apply(a, b, acier::minus());
}

template <class T, class U, size_t N, acier::when<std::is_scalar<U>::value> = true>
vec<std::common_type_t<T, U>, N> operator-(const vec<T, N>& a, const U& b) {
  using namespace std::placeholders;
  return apply(a, std::bind(acier::minus(), _1, b));
}

template <class T, class U, size_t N>
vec<T, N>& operator+=(vec<T, N>& a, const vec<U, N>& b) {
  gil::for_each(a, b, acier::plus_assign());
  return a;
}

template <class T, class U, size_t N>
vec<T, N>& operator+=(vec<T, N>& a, const U& b) {
  using namespace std::placeholders;
  gil::for_each(a, std::bind(acier::plus_assign(), _1, b));
  return a;
}

template <class T, class U, size_t N>
vec<T, N>& operator-=(vec<T, N>& a, const vec<U, N>& b) {
  gil::for_each(a, b, acier::minus_assign());
  return a;
}

template <class T, class U, size_t N>
vec<T, N>& operator-=(vec<T, N>& a, const U& b) {
  using namespace std::placeholders;
  gil::for_each(a, std::bind(acier::minus_assign(), _1, b));
  return a;
}

template <class T, class U, size_t N>
vec<std::common_type_t<T, U>, N> operator*(const vec<T, N>& a, const U& b) {
  using namespace std::placeholders;
  return apply(a, std::bind(acier::multiplies(), _1, b));
}

template <class T, class U, size_t N>
vec<std::common_type_t<T, U>, N> operator*(const T& a, const vec<U, N>& b) {
  return b * a;
}

template <class T, class U, size_t N>
vec<T, N>& operator*=(vec<T, N>& a, const U& b) {
  using namespace std::placeholders;
  gil::for_each(a, std::bind(acier::multiplies(), _1, b));
  return a;
}

template <class T, class U, size_t N>
vec<std::common_type_t<T, U>, N> operator/(const vec<T, N>& a, const U& b) {
  using namespace std::placeholders;
  return apply(a, std::bind(acier::divides(), _1, b));
}

template <class T, class U, size_t N>
vec<T, N>& operator/=(vec<T, N>& a, const U& b) {
  using namespace std::placeholders;
  gil::for_each(a, std::bind(acier::divides_assign(), _1, b));
  return a;
}

template <class T, class U, size_t N>
vec<T, N> saturate_cast(const vec<U, N>& x) {
  return apply(x, [](const U& v){ return acier::saturate<T>(v); });
}

template <class T> using vec2 = vec<T, 2>;
template <class T> using vec3 = vec<T, 3>;
template <class T> using vec4 = vec<T, 4>;

using vec3b = vec3<uint8_t>;
using vec3i = vec3<int>;
using vec3ui = vec3<unsigned>;
using vec3f = vec3<float>;

using vec4b = vec4<uint8_t>;
using vec4i = vec4<int>;
using vec4ui = vec4<unsigned>;
using vec4f = vec4<float>;

}
