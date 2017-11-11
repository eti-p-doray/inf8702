#pragma once

namespace cl {

#pragma once

#include "cl/wrapper.hpp"

namespace cl {

struct sampler_traits {
  using handle = cl_sampler;
  using info = 	cl_sampler_info;
  static constexpr auto get_info = clGetSamplerInfo;
  static constexpr auto retain = clRetainSampler;
  static constexpr auto release = clReleaseSampler;
};

class weak_sampler : public base_wrapper<sampler_traits> {
 public:
  using base_wrapper::base_wrapper;
  
 protected:
  void create(weak_context ctx, bool normalized_coords, addressing_mode addressing, filter_mode filter) {
    assert(ctx != nullptr);
    cl_int error = 0;
    cl_program id = clCreateSampler(
      ctx,
      normalized_coords,
      static_cast<cl_addressing_mode>(addressing),
      static_cast<>(filter),
      &error
    );
    if (!id) throw opencl_error(error);
    reset(id);
  }
};

enum class addressing_mode {
  kNone = CL_ADDRESS_NONE,
  kClampToEdge = CL_ADDRESS_CLAMP_TO_EDGE,
  kClamp = CL_ADDRESS_CLAMP,
  kRepeat = CL_ADDRESS_REPEAT,
};

enum class filter_mode {
  kNearest = CL_FILTER_NEAREST,
  kLinear = CL_FILTER_LINEAR,
};

struct sampler : shared_wrapper<weak_sampler> {
  using shared_wrapper::shared_wrapper;
  sampler() = default;
  sampler(weak_context ctx, bool normalized_coords, addressing_mode addressing, filter_mode filter) {
    create(normalized_coords, addressing, filter);
  }
};


}

}
