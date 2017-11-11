#include "cl/memory.hpp"

#include "cl/utility.hpp"

namespace cl {

buffer::buffer(weak_context ctx, size_t size, cl_mem_flags flags, void* host_ptr) {
  cl_int err = 0;
  cl_mem object = clCreateBuffer(ctx, flags, size, host_ptr, &err);
  if (!object) throw opencl_error(err);
  this->reset(object);
}

buffer::buffer(weak_context ctx, size_t size, bool host_memory,
         device_access da, host_access ha)
    : buffer(ctx, size, flags<cl_mem_flags>((host_memory ? CL_MEM_ALLOC_HOST_PTR : 0), da, ha), nullptr) {}
  
buffer::buffer(weak_context ctx, size_t size, void* host_ptr, bool host_memory,
         device_access da, host_access ha)
    : buffer(ctx, size, flags<cl_mem_flags>((host_memory ? CL_MEM_USE_HOST_PTR : CL_MEM_COPY_HOST_PTR), da, ha), host_ptr) {}

image::image(weak_context ctx, cl_mem_flags flags,
             const image_format& format,
             const image_desc& desc,
             void* host_ptr) {
  cl_int err = 0;
  cl_mem object = clCreateImage(ctx, flags, &format, &desc, host_ptr, &err);
  if (!object) throw opencl_error(err);
  this->reset(object);
}

image::image(weak_context ctx,
        const image_format& format,
        const image_desc& desc,
        bool memory_space,
        device_access da,
        host_access ha)
    : image(ctx, flags<cl_mem_flags>((memory_space ? CL_MEM_ALLOC_HOST_PTR : 0), da, ha), format, desc, nullptr) {}
  
image::image(weak_context ctx,
        const image_format& format,
        const image_desc& desc,
        void* host_ptr,
        bool memory_space,
        device_access da,
        host_access ha)
    : image(ctx, flags<cl_mem_flags>((memory_space ? CL_MEM_USE_HOST_PTR : CL_MEM_COPY_HOST_PTR), da, ha), format, desc, host_ptr) {}

}
