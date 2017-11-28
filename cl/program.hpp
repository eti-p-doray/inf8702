#pragma once

#include <string>
#include <vector>

#include "cl/wrapper.hpp"
#include "cl/context.hpp"
#include "cl/device.hpp"

namespace cl {

struct program_traits {
  using handle = cl_program;
  using info = 	cl_program_info;
  static constexpr auto get_info = clGetProgramInfo;
  static constexpr auto retain = clRetainProgram;
  static constexpr auto release = clReleaseProgram;
};

class weak_program : public base_wrapper<program_traits> {
  template <info i, class Type>
  using build_property = detail::property<cl_program_build_info, i, Type>;
 public:
  struct ReferenceCount : property<CL_PROGRAM_REFERENCE_COUNT, cl_uint> {};
  struct Context : property<CL_PROGRAM_CONTEXT, cl_context> {};
  struct NumDevices : property<CL_PROGRAM_NUM_DEVICES, cl_uint> {};
  struct Devices : property<CL_PROGRAM_DEVICES, std::vector<cl_device_id>> {};
  struct Source : property<CL_PROGRAM_SOURCE, std::string> {};
  struct BinarySizes : property<CL_PROGRAM_BINARY_SIZES, std::vector<size_t>> {};
  struct Binaries : property<CL_PROGRAM_BINARIES, std::vector<std::string>> {};
  
  struct BuildStatus : build_property<CL_PROGRAM_BUILD_STATUS, cl_build_status> {};
  struct BuildOptions : build_property<CL_PROGRAM_BUILD_OPTIONS, std::string> {};
  struct BuildLog : build_property<CL_PROGRAM_BUILD_LOG, std::string> {};
 
  using base_wrapper::base_wrapper;
  
  void build();
  
  template <class Property, class Type = typename Property::type>
  Type get_build_info(device d) const {
    Type info;
    detail::get_info_impl<Property>(clGetProgramBuildInfo, info, get(), d);
    return info;
  }
  
 protected:
  void create(weak_context ctx, const char* source);
  void create(weak_context ctx, const std::vector<std::string>& sources);
  void create(weak_context ctx, const std::vector<device>& devices, const std::vector<std::string>& binaries);
};

struct program : shared_wrapper<weak_program> {
  using shared_wrapper::shared_wrapper;
  program() = default;
  program(weak_context ctx, const char* source);
  program(weak_context ctx, const std::vector<std::string>& sources);
  program(weak_context ctx, const std::vector<device>& devices, const std::vector<std::string>& binaries);
};


}
