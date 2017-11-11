#pragma once

#include <assert.h>

#include <array>
#include <functional>
#include <memory>
#include <vector>

#include "gil/vec.hpp"

namespace gil {

template <class T>
class mat_view {
 public:
  using row_iterator = T*;
  using row_const_iterator = std::add_const_t<T>*;

  mat_view(vec2<size_t> size, size_t stride, T* data)
      : rows_(size[0]),
        cols_(size[1]),
        stride_(stride),
        data_(data) {}
  mat_view(const mat_view& that) = default;
  template <class U>
  mat_view(const mat_view<U>& that)
      : rows_(that.rows()),
        cols_(that.cols()),
        stride_(that.stride()),
        data_(that.data()) {}
  
  template <class U>
  mat_view& operator=(mat_view<U> that) {
    assert(size() == that.size());
    for (size_t i = 0; i < rows(); ++i) {
      std::copy(that.row_cbegin(i), that.row_cend(i), row_begin(i));
    }
    return *this;
  }
  mat_view& operator=(const T& value) {
    apply(std::bind(acier::identity(), value));
    return *this;
  }
  
  vec2<size_t> size() const { return {rows(), cols()}; }
  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  size_t total() const { return rows_ * cols_; }
  size_t stride() const { return stride_; }
  size_t pitch() const { return stride() * sizeof(T); }

  row_iterator row_begin(size_t row) const { return data_ + stride_ * row; }
  row_iterator row_end(size_t row) const { return data_ + stride_ * row + cols_; }
  row_const_iterator row_cbegin(size_t row) const { return data_ + stride_ * row; }
  row_const_iterator row_cend(size_t row) const { return data_ + stride_ * row + cols_; }
  
  T* data() const { return data_; }
  
  template <class F>
  mat_view& apply(const F& fcn) {
    using namespace std::placeholders;
    for (size_t i = 0; i < rows(); ++i) {
      std::for_each(row_begin(i), row_end(i), std::bind(acier::assign(), _1, std::bind(fcn, _1)));
    }
    return *this;
  }
  
  void swap(mat_view& other) {
    std::swap(rows_, other.rows_);
    std::swap(cols_, other.cols_);
    std::swap(stride_, other.stride_);
    std::swap(data_, other.data_);
  }

 private:
  size_t rows_;
  size_t cols_;
  size_t stride_;
  T* data_;
};

template <class T>
using mat_cview = mat_view<const T>;

template <class T, class Alloc = std::allocator<T>>
class mat {
 public:
  using row_iterator = typename std::vector<T, Alloc>::iterator;
  using row_const_iterator = typename std::vector<T, Alloc>::const_iterator;

  mat() = default;
  mat(const mat& that) = default;
  mat(mat&& that) = default;
  mat(vec2<size_t> size, const T& value = T(), const Alloc& alloc = Alloc())
      : rows_(size[0]),
        cols_(size[1]),
        data_(size[0] * size[1], value, alloc) {}
  mat(vec2<size_t> size, size_t pitch, const uint8_t* data, const Alloc& alloc = Alloc())
      : rows_(size[0]),
        cols_(size[1]),
        data_(size[0] * size[1], alloc) {
    auto d_first = data_.begin();
    for (size_t i = 0; i < rows_; ++i) {
      d_first = std::copy_n(reinterpret_cast<const T*>(data), cols_, d_first);
      data = data + pitch;
    }
  }
  template <class U>
  mat(mat_view<U> that, const Alloc& alloc = Alloc())
      : rows_(that.rows()),
        cols_(that.cols()),
        data_(that.rows() * that.cols(), alloc) {
    auto d_first = data_.begin();
    for (size_t i = 0; i < rows(); ++i) {
      d_first = std::copy(that.row_begin(i), that.row_end(i), d_first);
    }
  }
  template <class U>
  mat(const mat<U>& that, const Alloc& alloc = Alloc())
      : mat(gil::mat_cview<U>(that), alloc) {}
  
  mat& operator=(const mat& that) = default;
  mat& operator=(mat&& that) = default;
  
  mat& operator=(const T& value) {
    apply(std::bind(acier::identity(), value));
    return *this;
  }
  
  operator mat_view<const T>() const {
    return {{rows(), cols()}, cols(), data()};
  }
  operator mat_view<T>() {
    return {{rows(), cols()}, cols(), data()};
  }
  
  vec2<size_t> size() const { return {rows(), cols()}; }
  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  size_t total() const { return rows_ * cols_; }
  size_t stride() const { return cols_; }
  size_t pitch() const { return stride() * sizeof(T); }
  
  mat_view<T> operator[](const vec4<size_t>& r) {
    return {{r[2], r[3]}, stride(), data() + r[0] * stride() + r[1]};
  }
  mat_view<const T> operator[](const vec4<size_t>& r) const {
    return {r[2], r[3], stride(), data() + r[0] * stride() + r[1]};
  }

  row_iterator row_begin(size_t row) { return data_.begin() + cols() * row; }
  row_iterator row_end(size_t row) { return data_.begin() + cols() * (row+1); }
  row_const_iterator row_begin(size_t row) const { return data_.begin() + cols() * row; }
  row_const_iterator row_end(size_t row) const { return data_.begin() + cols() * (row+1); }
  row_const_iterator row_cbegin(size_t row) const { return row_cbegin(row); }
  row_const_iterator row_cend(size_t row) const { return row_cend(row); }
  
  const T* data() const { return data_.data(); }
  T* data() { return data_.data(); }
  
  template <class F>
  mat& apply(const F& fcn) {
    using namespace std::placeholders;
    for (size_t i = 0; i < rows(); ++i) {
      std::for_each(row_begin(i), row_end(i), std::bind(acier::assign(), _1, std::bind(fcn, _1)));
    }
    return *this;
  }
  
  void swap(mat& other) {
    std::swap(rows_, other.rows_);
    std::swap(cols_, other.cols_);
    std::swap(data_, other.data_);
  }

 private:
  size_t rows_;
  size_t cols_;
  std::vector<T, Alloc> data_;
};

template <class T, class U, class F>
void for_each(mat_view<T> a, mat_view<U> b, const F& fcn) {
  assert(a.size() == b.size());
  for (size_t i = 0; i < a.rows(); ++i) {
    auto a_it = a.row_begin(i);
    auto b_it = b.row_begin(i);
    for (size_t j = 0; j < a.cols(); ++j, ++a_it, ++b_it) {
      fcn(*a_it, *b_it);
    }
  }
}

template <class T, class U>
mat_view<T> operator+=(mat_view<T> a, mat_view<U> b) {
  for_each(a, b, acier::plus_assign());
  return a;
}

template <class T, class U, class Alloc1, class Alloc2>
mat<T, Alloc1>& operator+=(mat<T, Alloc1>& a, const mat<U, Alloc2>& b) {
  mat_view<T>(a) += mat_cview<U>(b);
  return a;
}

template <class T, class U>
mat_view<T> operator+=(mat_view<T> a, const U& b) {
  using namespace std::placeholders;
  a.apply(std::bind(acier::plus_assign(), _1, b));
  return a;
}

template <class T, class U, class Alloc1>
mat<T, Alloc1>& operator+=(mat<T, Alloc1>& a, const U& b) {
  mat_view<T>(a) += mat_cview<U>(b);
  return a;
}

template <class T, class U>
mat_view<T> operator-=(mat_view<T> a, mat_view<U> b) {
  for_each(a, b, acier::minus_assign());
  return a;
}

template <class T, class U, class Alloc1, class Alloc2>
mat<T, Alloc1>& operator-=(mat<T, Alloc1>& a, const mat<U, Alloc2>& b) {
  mat_view<T>(a) -= mat_cview<U>(b);
  return a;
}

template <class T, class U>
mat_view<T> operator-=(mat_view<T> a, const U& b) {
  using namespace std::placeholders;
  a.apply(std::bind(acier::minus_assign(), _1, b));
  return a;
}

template <class T, class U, class Alloc1>
mat<T, Alloc1>& operator-=(mat<T, Alloc1>& a, const U& b) {
  mat_view<T>(a) -= mat_cview<U>(b);
  return a;
}

}
