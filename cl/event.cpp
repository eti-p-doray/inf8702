#include "cl/event.hpp"

#include "cl/context.hpp"

namespace cl {

event::event(weak_context ctx) {
  cl_int err = 0;
  cl_event object = clCreateUserEvent(ctx, &err);
  if (!object) throw opencl_error(err);
  reset(object);
}

}
