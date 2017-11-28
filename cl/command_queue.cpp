#include "cl/command_queue.hpp"

#include "cl/context.hpp"

namespace cl {

command_queue::command_queue(weak_context ctx, device d,
              cl_command_queue_properties properties) {
  create(ctx, d, properties);
}

command_queue::command_queue(weak_context ctx,
              cl_command_queue_properties properties) {
  auto devices = ctx.devices();
  assert(!devices.empty());
  create(ctx, devices[0], properties);
}

void weak_command_queue::create(weak_context ctx, device d,
                cl_command_queue_properties properties) {
  assert(ctx != nullptr);
  assert(d != nullptr);
  cl_int error = 0;
  cl_command_queue id = clCreateCommandQueue(
    ctx,
    d,
    properties,
    &error
  );
  if (!id) throw opencl_error(error);
  reset(id);
}

}
