#pragma once

#include <string>
#include <vector>

#include "cl/cl.hpp"

namespace cl {

namespace detail {

template <class Info, Info v, class Type>
struct property {
  using info = Info;
  using type = Type;
  static constexpr info value = v;
};

template <class Info, Info v>
struct predicate : property<Info, v, cl_bool> {
  template <class Object>
  bool operator()(const Object& o) const {
    return o.template get_info<predicate>();
  }
};

template <class P, class F, class T, class... Args>
void get_info_impl(F fcn, T& info, const Args&... args) {
  fcn(args..., P::value, sizeof(T), reinterpret_cast<void*>(&info), nullptr);
}

template <class P, class F, class... Args>
void get_info_impl(F fcn, std::string& info, const Args&... args) {
  size_t size = 0;
  fcn(args..., P::value, 0, nullptr, &size);
  info.resize(size);
  fcn(args..., P::value, size, reinterpret_cast<void*>(&info[0]), nullptr);
}

template <class P, class F, class T, class... Args>
void get_info_impl(F fcn, std::vector<T>& info, const Args&... args) {
  size_t size = 0;
  fcn(args..., P::value, 0, nullptr, &size);
  info.resize(size / sizeof(T));
  fcn(args..., P::value, size, reinterpret_cast<void*>(info.data()), nullptr);
}

}

template <class Traits>
class base_wrapper {
 public:
  using traits = Traits;
  using handle = typename Traits::handle;
  using info = typename Traits::info;
 
  template <info i, class Type>
  using property = detail::property<info, i, Type>;
  template <info i>
  using predicate = detail::predicate<info, i>;
 
  base_wrapper() = default;
  base_wrapper(std::nullptr_t) {}
  base_wrapper(const base_wrapper& that) = default;
  base_wrapper& operator = (const base_wrapper& that) = default;
  
  const handle& get() const { return object_; }
  operator handle() const { return object_; }
  explicit operator bool() const { return object_ != nullptr; }
  
  void reset() {
    object_ = nullptr;
  }
  void reset(handle object) {
    object_ = object;
  }
  handle release() { return exchange(object_, nullptr); }
  
  void swap( base_wrapper& r ) {
    std::swap(object_, r.object_);
  }
  
  template <class Property, class Type = typename Property::type>
  Type get_info() const {
    Type info;
    detail::get_info_impl<Property>(Traits::get_info, info, object_);
    return info;
  }
  
  friend bool operator == (const base_wrapper& w, std::nullptr_t) { return !bool(w); }
  friend bool operator == (std::nullptr_t, const base_wrapper& w) { return !bool(w); }
  friend bool operator != (const base_wrapper& w, std::nullptr_t) { return bool(w); }
  friend bool operator != (std::nullptr_t, const base_wrapper& w) { return bool(w); }
  friend bool operator == (const base_wrapper& a, const base_wrapper& b) {
    return a.get() == b.get();
  }
  
 protected:
  base_wrapper(typename Traits::handle that) : object_(that) {}
  
 private:
  handle object_ = nullptr;
};

constexpr struct retain_t {} retain {};
constexpr struct transfer_t {} transfer {};

template <class Object>
class shared_wrapper : public Object {
 public:
  using typename Object::handle;
  using typename Object::traits;
  
  using Object::Object;
  shared_wrapper() = default;
  shared_wrapper(const shared_wrapper& that) : Object(that) {
    retain();
  }
  explicit shared_wrapper(const Object& that, retain_t) : Object(that) {
    retain();
  }
  explicit shared_wrapper(const Object& that, transfer_t) : Object(that) {}
  shared_wrapper(shared_wrapper&& that) : Object(that) {
    that.Object::reset();
  }
  shared_wrapper(typename traits::handle that, retain_t) : Object(that) {
    retain();
  }
  shared_wrapper(typename traits::handle that, transfer_t) : Object(that) {}
  shared_wrapper& operator = (const shared_wrapper& that) {
    if (this == &that) return *this;
    reset();
    Object::operator=(that);
    retain();
    return *this;
  }
  shared_wrapper& operator = (shared_wrapper&& that) {
    if (this == &that) return *this;
    reset();
    Object::operator=(that);
    that.Object::reset();
    return *this;
  }
  
  ~shared_wrapper() {
    reset();
  }
  
  void reset() {
    if (*this == nullptr) return;
    traits::release(this->get());
    Object::reset(nullptr);
  }
  void reset(handle object) {
    if (*this != nullptr)
      traits::release(this->get());
    Object::reset(object);
  }

 private:
  void retain() const {
    if (*this != nullptr)
      traits::retain(this->get());
  }

};

}
