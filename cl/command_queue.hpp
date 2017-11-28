#pragma once

#include <assert.h>

#include <initializer_list>
#include <vector>
#include <array>

#include "cl/wrapper.hpp"
#include "cl/device.hpp"
#include "cl/event.hpp"

namespace cl {

class weak_context;

struct command_queue_traits {
  using handle = cl_command_queue;
  using info = 	cl_command_queue_info;
  static constexpr auto get_info = clGetCommandQueueInfo;
  static constexpr auto retain = clRetainCommandQueue;
  static constexpr auto release = clReleaseCommandQueue;
};

using event_list = std::initializer_list<event>;

class weak_command_queue : public base_wrapper<command_queue_traits> {
 public:
  using base_wrapper<command_queue_traits>::base_wrapper;
  
  template <class R, class F, class... Args>
  future<R> enqueue(F fun, event_list el, Args&&... args) {
    cl_event e;
    cl_int err = 0;
    auto event_ptr = el.size() != 0 ? el.begin() : nullptr;
    R r = reinterpret_cast<R>(fun(*this, std::forward<Args>(args)...,
        cl_uint(el.size()),
        reinterpret_cast<const cl_event*>(event_ptr), &e, &err));
    if (err != CL_SUCCESS) throw opencl_error(err);
    return {r, e};
  }
  template <class F, class... Args>
  event enqueue(F fun, event_list el, Args&&... args) {
    cl_event e;
    auto event_ptr = el.size() != 0 ? el.begin() : nullptr;
    cl_int err = fun(*this, std::forward<Args>(args)...,
        cl_uint(el.size()),
        reinterpret_cast<const cl_event*>(event_ptr), &e);
    if (err != CL_SUCCESS) throw opencl_error(err);
    return {e, transfer};
  }
  
  void finnish();
  void flush();
  
 protected:
  void create(weak_context ctx, device d,
                cl_command_queue_properties properties = 0);
};

struct command_queue : shared_wrapper<weak_command_queue> {
  using shared_wrapper::shared_wrapper;
  command_queue() = default;
  command_queue(weak_context ctx, device d,
                cl_command_queue_properties properties = 0);
  command_queue(weak_context ctx,
                cl_command_queue_properties properties = 0);
};



}
