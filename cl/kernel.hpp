#pragma once

#include <tuple>
#include <string>
#include <initializer_list>

#include "cl/wrapper.hpp"
#include "cl/program.hpp"
#include "cl/memory.hpp"

namespace cl {

struct kernel_traits {
  using handle = cl_kernel;
  using info = 	cl_kernel_info;
  static constexpr auto get_info = clGetKernelInfo;
  static constexpr auto retain = clRetainKernel;
  static constexpr auto release = clReleaseKernel;
};

class weak_kernel : public base_wrapper<kernel_traits> {
 public:
  struct FunctionName : property<CL_KERNEL_FUNCTION_NAME, std::string> {};
  struct NumArgs : property<CL_KERNEL_NUM_ARGS, cl_uint> {};
  struct ReferenceCount : property<CL_KERNEL_REFERENCE_COUNT, cl_uint> {};
  struct Context : property<CL_KERNEL_CONTEXT, cl_context> {};
  struct Program : property<CL_KERNEL_PROGRAM, cl_program> {};
 
  using base_wrapper::base_wrapper;
  
  //template <class... Args>
  //event call(command_queue queue, )
  
 protected:
  void create(weak_program prog, const char* name);
};

struct kernel : shared_wrapper<weak_kernel> {
  using shared_wrapper::shared_wrapper;
  
  kernel() = default;
  kernel(weak_program prog, const char* name);
  kernel(weak_program prog, const std::string& name) : kernel(prog, name.c_str()) {}
};

std::vector<kernel> create_kernels(weak_program prog, const std::vector<std::string>& names);

namespace detail {

template <class T>
void set_kernel_arg(weak_kernel k, size_t index, const T& v) {
  cl_int error = clSetKernelArg(k, cl_uint(index), sizeof(v), &v);
  check_error(error);
}

inline void set_kernel_arg(weak_kernel k, size_t index, weak_buffer buf) {
  cl_int error = clSetKernelArg(k, cl_uint(index), sizeof(cl_mem), buf.get());
  check_error(error);
}

template <size_t... I, class... Args>
void set_kernel_args(weak_kernel k,  std::index_sequence<I...>, const std::tuple<Args...>& args) {
  auto dummy = {(set_kernel_arg(k, I, std::get<I>(args)), 0)...};
  (void)dummy;
}

}

template <class... Args>
auto invoke_kernel(weak_kernel k,
                   std::initializer_list<size_t> global_offsets,
                   std::initializer_list<size_t> global_size,
                   std::initializer_list<size_t> local_size,
                   const std::tuple<Args...>& args) {
  detail::set_kernel_args(k, std::index_sequence_for<Args...>(), args);
  return [k, global_offsets, global_size, local_size](weak_command_queue q, event_list el) -> event {
    return q.enqueue(clEnqueueNDRangeKernel, el, k, global_size.size(), global_offsets.begin(), global_size.begin(), local_size.begin());
  };
}

template <class... Args>
auto invoke_kernel(weak_kernel k,
                   std::initializer_list<size_t> global_offsets,
                   std::initializer_list<size_t> global_size,
                   const std::tuple<Args...>& args) {
  detail::set_kernel_args(k, std::index_sequence_for<Args...>(), args);
  return [k, global_offsets, global_size](weak_command_queue q, event_list el) -> event {
    return q.enqueue(clEnqueueNDRangeKernel, el, k, global_size.size(), global_offsets.begin(), global_size.begin(), nullptr);
  };
}

template <class... Args>
auto invoke_kernel(weak_kernel k,
                   std::initializer_list<size_t> global_size,
                   const std::tuple<Args...>& args) {
  detail::set_kernel_args(k, std::index_sequence_for<Args...>(), args);
  return [k, global_size](weak_command_queue q, event_list el) -> event {
    return q.enqueue(clEnqueueNDRangeKernel, el, k, cl_uint(global_size.size()), nullptr, global_size.begin(), nullptr);
  };
}

}
