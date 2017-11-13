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
  
  template <class F, acier::when<ptr_callback<F>()> = true>
  void bind(const F& callback, cl_int status = complete);
  template <class F, acier::when<!ptr_callback<F>()> = true>
  void bind(const F& callback, cl_int status = complete);
  void bind(native_function callback, void* user_data, cl_int status = complete);
};

struct event : shared_wrapper<weak_event> {
  using shared_wrapper::shared_wrapper;
  using shared_wrapper::operator=;
  event() = default;
  event(weak_context ctx);
};

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

}


