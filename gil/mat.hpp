#pragma once

#include <assert.h>

#include <array>
#include <functional>
#include <memory>
#include <vector>

#include <opencv2/core/mat.hpp>

#include "acier/compressed_member.hpp"
#include "gil/vec.hpp"

namespace gil {

template <class T>
struct cv_channel {};

template <>
struct cv_channel<uint8_t> { static constexpr int value = CV_8U; };
template <>
struct cv_channel<gil::vec3b> { static constexpr int value = CV_8UC3; };
template <>
struct cv_channel<gil::vec4b> { static constexpr int value = CV_8UC4; };
template <>
struct cv_channel<gil::vec3f> { static constexpr int value = CV_32FC3; };
template <>
struct cv_channel<gil::vec4f> { static constexpr int value = CV_32FC4; };

template <class T>
class mat_view {
 public:
  using row_iterator = T*;
  using row_const_iterator = std::add_const_t<T>*;

  mat_view(cv::Mat that)
      : rows_(that.rows),
        cols_(that.cols),
        stride_(that.step1() / that.channels()),
        data_(that.ptr<T>(0)) {
    assert(that.type() == cv_channel<T>::value);
  }
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
  
  operator cv::Mat() const {
    return cv::Mat(int(rows()), int(cols()), cv_channel<T>::value,
                   reinterpret_cast<void*>(data()), pitch());
  }
  
  vec2<size_t> size() const { return {rows(), cols()}; }
  size_t rows() const { return rows_; }
  size_t cols() const { return cols_; }
  size_t total() const { return rows_ * cols_; }
  size_t stride() const { return stride_; }
  size_t pitch() const { return stride() * sizeof(T); }
  
  mat_view operator[](gil::vec4<size_t> frame) const {
    return {{frame[2], frame[3]}, stride(), row_begin(frame[0]) + frame[1]};
  }

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
  
  explicit operator bool() const { return data_ != nullptr; }
  
  friend bool operator == (const mat_view& m, std::nullptr_t) { return !bool(m); }
  friend bool operator == (std::nullptr_t, const mat_view& m) { return !bool(m); }
  friend bool operator != (const mat_view& m, std::nullptr_t) { return bool(m); }
  friend bool operator != (std::nullptr_t, const mat_view& m) { return bool(m); }

 protected:
  void reset() {
    rows_ = 0;
    cols_ = 0;
    stride_ = 0;
    data_ = nullptr;
  }
  
  size_t rows_ = 0;
  size_t cols_ = 0;
  size_t stride_ = 0;
  T* data_ = nullptr;
};

template <class T>
using mat_cview = mat_view<const T>;

constexpr struct retain_t {} retain {};

template <class T, class Alloc = std::allocator<T>>
class mat : public mat_view<T>,
            private acier::compressed_member<Alloc> {
 public:
  using row_iterator = typename mat_view<T>::row_iterator;
  using row_const_iterator = typename mat_view<T>::row_const_iterator;

  mat() = default;
  mat(const mat& that)
      : mat(that.size()) {
    for (size_t i = 0; i < this->rows(); ++i) {
      std::copy(that.row_cbegin(i), that.row_cend(i), this->row_begin(i));
    }
  }
  mat(mat&& that)
      : mat_view<T>(that),
        acier::compressed_member<Alloc>(std::move(that.get_alloc())) {
    that.mat_view<T>::reset();
  }
  explicit mat(const mat_view<T>& that, retain_t, const Alloc& alloc = Alloc())
      : mat_view<T>(that),
        acier::compressed_member<Alloc>(alloc) {}
  mat(vec2<size_t> size, const T& value = T(), const Alloc& alloc = Alloc())
      : mat_view<T>(size, size[1], nullptr) {
    this->data_ = std::allocator_traits<Alloc>::allocate(get_alloc(), size[0] * size[1]);
    std::fill(this->data(), this->data() + this->total(), value);
  }
  template <class U>
  mat(vec2<size_t> size, size_t pitch, const U* data, const Alloc& alloc = Alloc())
      : mat_view<T>(size, size[1], nullptr),
        acier::compressed_member<Alloc>(alloc) {
    this->data_ = std::allocator_traits<Alloc>::allocate(get_alloc(), size[0] * size[1]);
    auto d_first = this->data();
    auto ptr = reinterpret_cast<const uint8_t*>(data);
    for (size_t i = 0; i < this->rows(); ++i) {
      d_first = std::copy_n(reinterpret_cast<const U*>(ptr), this->cols(), d_first);
      data = data + pitch;
    }
  }
  template <class U>
  mat(mat_view<U> that, const Alloc& alloc = Alloc())
      : mat_view<T>(that.size(), that.cols(), nullptr),
        acier::compressed_member<Alloc>(alloc) {
    this->data_ = std::allocator_traits<Alloc>::allocate(get_alloc(), this->total());
    auto d_first = this->data();
    for (size_t i = 0; i < this->rows(); ++i) {
      d_first = std::copy(that.row_begin(i), that.row_end(i), d_first);
    }
  }
  template <class U>
  mat(const mat<U>& that, const Alloc& alloc = Alloc())
      : mat(gil::mat_cview<U>(that), alloc) {}
  
  mat& operator=(const mat& that);
  mat& operator=(mat&& that) {
    mat_view<T>::operator=(that);
    that.reset();
  }
  
  ~mat() {
    destroy();
  }
  
  mat& operator=(const T& value) {
    this->apply(std::bind(acier::identity(), value));
    return *this;
  }
  
  row_iterator row_begin(size_t row) { return this->data() + this->stride() * row; }
  row_iterator row_end(size_t row) { return this->data() + this->stride() * row + this->cols(); }
  row_const_iterator row_begin(size_t row) const { return this->data() + this->stride() * row; }
  row_const_iterator row_end(size_t row) const { return this->data() + this->stride() * row + this->cols(); }
  
  void swap(mat& other) {
    mat_view<T>::swap(other);
    std::swap(get_alloc(), other.get_alloc());
  }
  
  void reset() {
    if (*this == nullptr) return;
    destroy();
    mat_view<T>::reset();
  }

 private:
  void destroy() {
    std::allocator_traits<Alloc>::deallocate(get_alloc(), this->data(), this->total());
  }
  
  Alloc& get_alloc() { return acier::compressed_member<Alloc>::get(); }
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
