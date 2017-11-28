#pragma once

#include <assert.h>

#include <vector>

#include "acier/algorithm.hpp"
#include "cl/wrapper.hpp"
#include "cl/platform.hpp"

namespace cl {

struct device_traits {
  using handle = cl_device_id;
  using info = cl_device_info;
  static constexpr auto get_info = clGetDeviceInfo;
};

class device : public base_wrapper<device_traits> {
 public:
  struct AddressBits : property<CL_DEVICE_ADDRESS_BITS, cl_uint> {};
  struct Available : predicate<CL_DEVICE_AVAILABLE> {};
  struct CompilerAvailable : predicate<CL_DEVICE_COMPILER_AVAILABLE> {};
  struct DoubleFpConfig : property<CL_DEVICE_DOUBLE_FP_CONFIG, cl_device_fp_config> {};
  struct EndianLittle : predicate<CL_DEVICE_ENDIAN_LITTLE> {};
  struct ErrorCorrectionSupport : predicate<CL_DEVICE_ERROR_CORRECTION_SUPPORT> {};
  struct ExecutionCapabilities : property<CL_DEVICE_EXECUTION_CAPABILITIES, cl_device_exec_capabilities> {};
  struct Extensions : property<CL_DEVICE_EXTENSIONS, std::string> {};
  struct GlobalMemCacheSize : property<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, cl_ulong> {};
  struct GlobalMemCacheType : property<CL_DEVICE_GLOBAL_MEM_CACHE_TYPE, cl_device_mem_cache_type> {};
  struct GlobalMemCachelineSize : property<CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE, cl_uint> {};
  struct GlobalMemSize : property<CL_DEVICE_GLOBAL_MEM_SIZE, cl_ulong> {};
  struct HalfFpConfig : property<CL_DEVICE_HALF_FP_CONFIG, cl_device_fp_config> {};
  struct ImageSupport : predicate<CL_DEVICE_IMAGE_SUPPORT> {};
  
  struct Name : property<CL_DEVICE_NAME, std::string> {};
  
  struct Type : property<CL_DEVICE_TYPE, cl_device_type> {};
  
  using base_wrapper<device_traits>::base_wrapper;
  device() = default;
  device(cl_device_id object) : base_wrapper(object) {}
  
  bool available() const { return get_info<Available>(); }
  std::string name() const { return get_info<Name>(); }
  cl_device_type type() const { return get_info<Type>(); }
};

template <class Os>
Os& operator << (Os& os, const device& d) {
  os << d.name();
}

namespace filter {

struct gpu {
  bool operator()(const device& d) const
  { return d.type() == CL_DEVICE_TYPE_GPU; }
};

struct accelerator {
  bool operator()(const device& d) const
  { return d.type() == CL_DEVICE_TYPE_ACCELERATOR; }
};

struct cpu {
  bool operator()(const device& d) const
  { return d.type() == CL_DEVICE_TYPE_CPU; }
};

class count {
 public:
  count(size_t n) : n_(n) {}
  bool operator()(const device& d) { return i_++ < n_; }
 private:
  size_t i_ = 0;
  const size_t n_;
};

struct one : count {
  one() : count(1) {}
};

}

template <class... Filters>
std::vector<device> get_devices(Filters&&... filters) {
  std::vector<platform> platforms = get_platforms();
  
  std::vector<device> devices;
  for (auto& p : platforms) {
    cl_uint size = 0;
    clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, 0, nullptr, &size);
    std::vector<cl_device_id> device_ids(size);
    clGetDeviceIDs(p, CL_DEVICE_TYPE_ALL, size, device_ids.data(), nullptr);
    for (auto& id : device_ids) {
      device d(id);
      if (device::Available()(d) &&
          acier::all_of(std::initializer_list<bool>{filters(d)...})) {
          devices.push_back(std::move(d));
      }
    }
  }
  return devices;
}

inline device find_default_device() {
  auto d = get_devices(filter::gpu(), filter::one());
  if (d.empty())
    d = get_devices(filter::one());
  assert(!d.empty());
  return d.front();
}

inline device& default_device() {
  static device d = find_default_device();
  return d;
}

}
