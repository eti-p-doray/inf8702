#include "cl/kernel.hpp"

namespace cl {

void weak_kernel::create(weak_program prog, const char* name) {
  assert(prog != nullptr);
  cl_int error = 0;
  cl_kernel id = clCreateKernel(prog, name, &error);
  if (!id) throw opencl_error(error);
  reset(id);
}

kernel::kernel(weak_program prog, const char* name) {
  create(prog, name);
}

std::vector<kernel> create_kernels(weak_program prog, const std::vector<std::string>& names) {
  std::vector<cl_kernel> cl_kernels(names.size());
  cl_uint num_kernels_ret;
  cl_int error = clCreateKernelsInProgram(prog, cl_uint(names.size()), cl_kernels.data(), &num_kernels_ret);
  check_error(error);
  
  std::vector<kernel> kernels(num_kernels_ret);
  for (unsigned i = 0; i < num_kernels_ret; ++i) {
    kernels.push_back({cl_kernels[i], retain});
  }
  return kernels;
}

}
