#pragma once

#include <assert.h>

#include "acier/type_traits.hpp"
#include "cl/wrapper.hpp"
#include "cl/device.hpp"
#include "cl/error.hpp"

namespace cl {

class weak_context;

struct event_traits {
  using handle = cl_event;
  using info = 		cl_event_info;
  static constexpr auto get_info = clGetEventInfo;
  static constexpr auto retain = clRetainEvent;
  static constexpr auto release = clReleaseEvent;
};

class weak_event : public base_wrapper<event_traits> {
  template <class F, class D = decltype(reinterpret_cast<void(*)(weak_event, cl_int)>(std::declval<F>()))>
  static constexpr bool ptr_callback() { return true; }
  template <class F, class... D>
  static constexpr bool ptr_callback(D...) { return false; }

 public:
  using native_function = void(CL_CALLBACK*)(cl_event, cl_int, void*);
  enum execution_status {
    complete = CL_COMPLETE,
    running = CL_RUNNING,
    submitted = CL_SUBMITTED,
    queued = CL_QUEUED
  };
 
  using base_wrapper::base_wrapper;
  
  void notify(cl_int status = complete) {
    cl_int err = clSetUserEventStatus(*this, status);
    if (err != CL_SUCCESS) throw opencl_error(err);
  }
  void wait() {
    cl_int err = clWaitForEvents(1, &get());
    if (err != CL_SUCCESS) throw opencl_error(err);
  }
  template <class Container>
  static void wait(const Container&& c);
  
  template <class F, acier::when<ptr_callback<F>()> = true>
  void bind(const F& callback, cl_int status = complete);
  template <class F, acier::when<!ptr_callback<F>()> = true>
  void bind(const F& callback, cl_int status = complete);
  void bind(native_function callback, void* user_data, cl_int status = complete);
};

struct event : shared_wrapper<weak_event> {
  using shared_wrapper::shared_wrapper;
  event() = default;
  event(weak_context ctx);
};

/*template <class T>
void event::bind(const T& callback, cl_int status = CL_COMPLETE) {
  T* c = new T(callback);
  auto f = [](cl_event e, cl_int status, void* data) {
    T* c = reinterpret_cast<T*>(data);
    event e(e);
    c(e, status);
    e.release();
    delete c;
  }
  bind(f, status);
}

template <class T>
void event::bind(const T& callback, cl_int status = CL_COMPLETE) {
  auto f = [](cl_event e, cl_int status, void* data) {
    T c = reinterpret_cast<T>(data);
    event e(e);
    c(e, status);
    e.release();
  }
  bind(f, status);
}*/

/*namespace detail {

template <class T>
struct future_value {
  boost::variant<T, std::exception_ptr> value;
  event e;
  
  future_value(event&& e2) : e(std::move(e2)) {}

  template <class U>
  void set(U&& v) { *boost::get<T>(&value) = std::move(v); }
  void set_exception(std::exception_ptr p) { *boost::get<std::exception_ptr>(&value) = p; }
  T get() const {
    if (value.which() != 0)
      std::rethrow_exception(boost::get<std::exception_ptr>(value));
    return boost::get<T>(value);
  }
  
};

}*/

template <class T>
class future {
 public:
  future() = default;
  future(T value, weak_event e)
      : value_(value),
        event_(e, retain) {}
 
  T get() {
    wait();
    return value_;
  }
  void wait() {
    event_.wait();
  }
  
  weak_event get_event() { return value_->e_; }
  
  operator T() { return get(); }
  operator weak_event() { return get_event(); }
  
 private:
  T value_;
  event event_;
};

/*template <class T>
class unique_future {
 public:
  future() = default;
  future(std::unique_ptr<detail::future_value<T>>&& v)
      : value_(std::move(v)) {}
 
  T get() {
    wait();
    return value_->get();
  }
  void wait() { value_->e.wait(); }
  
  weak_event get_event() { return value_->e_; }
  
  operator T() { return get(); }
  operator weak_event() { return get_event(); }
  
 private:
  std::unique_ptr<detail::future_value<T>> value_;
};

template <class T>
class promise {
 public:
  future<T> get_future(weak_context c) {
    auto v = make_unique<detail::future_value<T>>(event(c));
    future_value_ = v.get();
    return future<T>(std::move(v));
  }
  template <class U>
  void set_value(U&& v) {
    future_value_->set(std::forward<U>(v));
    future_value_->e.notify();
  }
  void set_exception(std::exception_ptr p) {
    future_value_->set_exception(p);
    future_value_->e.notify();
  }
  
 private:
  detail::future_value<T>* future_value_;
};*/


}


