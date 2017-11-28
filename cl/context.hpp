#pragma once

#include <assert.h>

#include <vector>

#include "wrapper.hpp"
#include "device.hpp"
#include "command_queue.hpp"

namespace cl {

struct context_traits {
  using handle = cl_context;
  using info = cl_context_info;
  static constexpr auto get_info = clGetContextInfo;
  static constexpr auto retain = clRetainContext;
  static constexpr auto release = clReleaseContext;
};

class weak_context : public base_wrapper<context_traits> {
 public:
  struct ReferenceCount : property<CL_CONTEXT_REFERENCE_COUNT, cl_uint> {};
  struct Devices : property<CL_CONTEXT_DEVICES, std::vector<device>> {};
  struct Properties : property<CL_CONTEXT_PROPERTIES, cl_context_properties> {};
  
  using base_wrapper::base_wrapper;
  
  std::vector<device> devices() const { return get_info<Devices>(); }
  device default_device() const { return devices().front(); }
  command_queue& default_queue() {
    if (!default_queue_) {
      default_queue_ = command_queue(*this, default_device());
    }
    return default_queue_;
  }
  
 private:
  command_queue default_queue_;
};
struct context : shared_wrapper<weak_context> {
  using shared_wrapper::shared_wrapper;
  
  context() = default;
  context(const std::vector<device>& devices) {
    assert(!devices.empty());
    cl_int error = 0;
    cl_context id = clCreateContext(
      nullptr,
      cl_uint(devices.size()),
      reinterpret_cast<const cl_device_id*>(devices.data()),
      nullptr, nullptr, &error);
    if (!id) throw opencl_error(error);
    reset(id);
  }
  context(device d) {
    assert(d);
    cl_int error = 0;
    cl_context id = clCreateContext(
      nullptr,
      cl_uint(1),
      reinterpret_cast<const cl_device_id*>(&d),
      nullptr, nullptr, &error);
    if (!id) throw opencl_error(error);
    reset(id);
  }
};

inline weak_context default_context() {
  static context ctx(default_device());
  return ctx;
}


template <class Os>
Os& operator << (Os& os, const context& d) {
  auto devices = d.devices();
  for (size_t i = 0; i < devices.size(); ++i) {
    os << i << ". " << devices[i].name() << std::endl;
  }
  return os;
}



}
