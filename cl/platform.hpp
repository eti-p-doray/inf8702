#pragma once

#include <vector>

#include "cl/cl.hpp"
#include "cl/wrapper.hpp"

namespace cl {

struct platform_traits {
  using handle = cl_platform_id;
  using info = cl_platform_info;
  static constexpr auto get_info = clGetPlatformInfo;
};

class platform : public base_wrapper<platform_traits> {
 public:
  struct profile : property<CL_PLATFORM_PROFILE, std::string> {};
 
  using base_wrapper<platform_traits>::base_wrapper;
  platform() = default;
  platform(cl_platform_id object) : base_wrapper(object) {}
};

std::vector<platform> get_platforms();

}
