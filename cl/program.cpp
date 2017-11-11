#include "cl/program.hpp"

namespace cl {

void weak_program::build() {
  cl_int error = clBuildProgram(*this, 0, nullptr, "-cl-fast-relaxed-math", nullptr, nullptr);
  check_error(error);
}

void weak_program::create(weak_context ctx, const char* source) {
  assert(ctx != nullptr);
  assert(source != nullptr);
  cl_int error = 0;
  size_t length = strlen(source);
  cl_program id = clCreateProgramWithSource(
    ctx,
    1,
    &source,
    &length,
    &error
  );
  if (!id) throw opencl_error(error);
  reset(id);
}

program::program(weak_context ctx, const char* source) {
  create(ctx, source);
}

}
