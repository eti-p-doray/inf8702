#pragma once

#include <assert.h>

#include "cl/wrapper.hpp"
#include "cl/device.hpp"
#include "cl/context.hpp"

namespace cl {

struct memory_traits {
  using handle = cl_mem;
  using info = 	cl_mem_info;
  static constexpr auto get_info = clGetMemObjectInfo;
  static constexpr auto retain = clRetainMemObject;
  static constexpr auto release = clReleaseMemObject;
};

enum class device_access {
  kReadWrite = CL_MEM_READ_WRITE,
  kRead = CL_MEM_READ_ONLY,
  kWrite = CL_MEM_WRITE_ONLY,
};
enum class host_access {
  kNoAccess = CL_MEM_HOST_NO_ACCESS,
  kReadWrite = 0,
  kRead = CL_MEM_HOST_READ_ONLY,
  kWrite = CL_MEM_HOST_WRITE_ONLY,
};

class weak_memory : public base_wrapper<memory_traits> {
 public:
  using base_wrapper::base_wrapper;

  static constexpr bool host = true;
  static constexpr bool device = false;
};

struct memory : shared_wrapper<weak_memory> {
  using shared_wrapper::shared_wrapper;
};

class weak_buffer : public weak_memory {
 public:
  using weak_memory::weak_memory;
};

struct buffer : shared_wrapper<weak_buffer> {
  using shared_wrapper::shared_wrapper;
  buffer() = default;
  buffer(weak_context ctx, size_t size, cl_mem_flags flags, void* host_ptr = nullptr);
  buffer(weak_context ctx, size_t size, bool memory_space = host,
         device_access da = device_access::kReadWrite,
         host_access ha = host_access::kReadWrite);
  buffer(weak_context ctx, size_t size, void* host_ptr, bool memory_space = host,
         device_access da = device_access::kReadWrite,
         host_access ha = host_access::kReadWrite);
};

class weak_image : public weak_memory {
 public:
  using weak_memory::weak_memory;
};

enum class channel_order {
  kR = CL_R,
  kRx = CL_Rx,
  kA = CL_A,
  kIntensity = CL_INTENSITY,
  kLuminance = CL_LUMINANCE,
  kRG = CL_RG,
  kRGx = CL_RGx,
  kRA = CL_RA,
  kRGB = CL_RGB,
  kRGBx = CL_RGBx,
  kRGBA = CL_RGBA,
  kARGB = CL_ARGB,
  kBGRA = CL_BGRA
};

enum class channel_type {
  kSNormInt8 = CL_SNORM_INT8,
  kSNormInt16 = CL_SNORM_INT16,
  kUNormInt8 = CL_UNORM_INT8,
  kUNormInt16 = CL_UNORM_INT16,
  kUNormShort565 = CL_UNORM_SHORT_565,
  kUNormShort555 = CL_UNORM_SHORT_555,
  kUNormShort101010 = CL_UNORM_INT_101010,
  kSInt8 = CL_SIGNED_INT8,
  kSInt16 = CL_SIGNED_INT16,
  kSInt32 = CL_SIGNED_INT32,
  kUInt8 = CL_UNSIGNED_INT8,
  kUInt16 = CL_UNSIGNED_INT16,
  kUInt32 = CL_UNSIGNED_INT32,
  kHalfFloat = CL_HALF_FLOAT,
  kFloat = CL_FLOAT,
};

enum class mem_object_type {

};

struct image_format : cl_image_format {
  image_format(channel_order order, channel_type type)
      : cl_image_format{static_cast<cl_channel_order>(order),
                        static_cast<cl_channel_type>(type)} {}
};

using image_desc = cl_image_desc;
  //image_desc(mem_object_type image_type, size_t width, size_t height, size_t depth, )
//};

struct image : shared_wrapper<weak_image> {
  using shared_wrapper::shared_wrapper;
  image() = default;
  image(weak_context ctx, cl_mem_flags flags,
        const image_format& format,
        const image_desc& desc,
        void* host_ptr = nullptr);
  image(weak_context ctx,
        const image_format& format,
        const image_desc& desc,
        bool memory_space = host,
        device_access da = device_access::kReadWrite,
        host_access ha = host_access::kReadWrite);
  image(weak_context ctx,
        const image_format& format,
        const image_desc& desc,
        void* host_ptr,
        bool memory_space = host,
        device_access da = device_access::kReadWrite,
        host_access ha = host_access::kReadWrite);
};

enum class map_access {
  read = CL_MAP_READ,
  write = CL_MAP_WRITE,
  read_write = CL_MAP_READ | CL_MAP_WRITE,
  write_exclusive = CL_MAP_WRITE_INVALIDATE_REGION,
};

template <class T>
auto map_buffer(weak_buffer b, map_access access, size_t offset, size_t size) {
  return [b, access, offset, size](weak_command_queue q, event_list el){
    return q.enqueue(clEnqueueMapBuffer, el, b, CL_FALSE, access, offset * sizeof(T), size * sizeof(T));
  };
}

template <class T>
auto read_buffer(weak_buffer b, size_t offset, size_t size, T* ptr) {
  return [b, offset, size, ptr](weak_command_queue q, event_list el){
    return q.enqueue(clEnqueueReadBuffer, el, b, CL_FALSE, offset * sizeof(T), size * sizeof(T), reinterpret_cast<void*>(ptr));
  };
}

template <class T>
auto write_buffer(weak_buffer b, size_t offset, size_t size, const T* ptr) {
  return [b, offset, size, ptr](weak_command_queue q, event_list el){
    return q.enqueue(clEnqueueWriteBuffer, el, b, CL_FALSE, offset * sizeof(T), size * sizeof(T), reinterpret_cast<const void*>(ptr));
  };
}

template <class T>
auto fill_buffer(weak_buffer b, const T& value, size_t offset, size_t size) {
  return [b, offset, size, value](weak_command_queue q, event_list el){
    return q.enqueue(clEnqueueFillBuffer, el, b, reinterpret_cast<const void*>(&value), sizeof(T), offset * sizeof(T), size * sizeof(T));
  };
}

template <class T>
auto copy_buffer(weak_buffer src, weak_buffer dst, size_t src_offset, size_t dst_offset, size_t size) {
  return [src, dst, src_offset, dst_offset, size](weak_command_queue q, event_list el){
    return q.enqueue(clEnqueueCopyBuffer, el, src, dst, src_offset * sizeof(T), dst_offset * sizeof(T), size * sizeof(T));
  };
}

inline auto read_image(weak_image im, std::initializer_list<size_t> origin, std::initializer_list<size_t> region, size_t pitch, uint8_t* data) {
  return [im, origin, region, pitch, data](weak_command_queue q, event_list el){
    return q.enqueue(clEnqueueReadImage, el, im, CL_FALSE, origin.begin(), region.begin(), pitch, 0, data);
  };
}

inline auto write_image(weak_image im, std::initializer_list<size_t> origin, std::initializer_list<size_t> region, size_t pitch, const uint8_t* data) {
  return [im, origin, region, pitch, data](weak_command_queue q, event_list el){
    return q.enqueue(clEnqueueWriteImage, el, im, CL_FALSE, origin.begin(), region.begin(), pitch, 0, data);
  };
}

template <class T>
inline auto fill_image(weak_image im, const T& color,
    std::initializer_list<size_t> origin,
    std::initializer_list<size_t> region) {
  return [im, color, origin, region](weak_command_queue q, event_list el){
    return q.enqueue(clEnqueueFillImage, el, im,
      reinterpret_cast<const void*>(&color),
      origin.begin(), region.begin());
  };
}

}
