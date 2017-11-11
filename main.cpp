
#include <chrono>
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include <opencv2/highgui/highgui.hpp>

#include "gil/mat.hpp"
#include "gil/vec.hpp"
#include "poisson.hpp"

#include "cl/device.hpp"
#include "cl/context.hpp"
#include "cl/memory.hpp"
#include "cl/program.hpp"
#include "cl/kernel.hpp"

boost::filesystem::path find_file(const std::string& filename) {
  boost::filesystem::path path(__FILE__);
  path.remove_filename();
  path.append("poisson.cl");
  return path;
}

int main() {
  using namespace std::placeholders;
  
  auto device = cl::get_devices(cl::filter::gpu())[0];
  cl::context ctx(device);
  std::cout << ctx << std::endl;
  
  boost::iostreams::mapped_file_source poisson_source(find_file("poisson.cl"));
  
  cl::program prog(ctx, poisson_source.data());
  try {
    prog.build();
  } catch (...) {
    std::cout << prog.get_build_info<cl::program::BuildLog>(device);
    return 0;
  }
  cl::kernel make_boundary(prog, "make_boundary");
  cl::kernel make_guidance(prog, "make_guidance");
  cl::kernel jacobi_iteration(prog, "jacobi_iteration");
  cl::kernel apply_mask(prog, "apply_mask");
  
  gil::mat<uint8_t> mask(gil::mat_view<gil::vec3b>(cv::imread("mask.jpg")));
  gil::mat<gil::vec4f> src(gil::mat_view<gil::vec3b>(cv::imread("src.jpg", CV_LOAD_IMAGE_COLOR)));
  gil::mat<gil::vec4f> dst(gil::mat_view<gil::vec3b>(cv::imread("dst.jpg", CV_LOAD_IMAGE_COLOR)));
  
  cl::image g_mask(ctx,
    cl::image_format{cl::channel_order::kR, cl::channel_type::kUInt8},
    cl::image_desc{CL_MEM_OBJECT_IMAGE2D, mask.cols(), mask.rows(),
                   0, 1, 0, 0, 0, 0, nullptr},
    cl::buffer::device);
  
  cl::image g_boundary(ctx,
    cl::image_format{cl::channel_order::kR, cl::channel_type::kUInt8},
    cl::image_desc{CL_MEM_OBJECT_IMAGE2D, mask.cols(), mask.rows(),
                   0, 1, 0, 0, 0, 0, nullptr},
    cl::buffer::device);
  
  cl::image g_f(ctx,
    cl::image_format{cl::channel_order::kRGBA, cl::channel_type::kFloat},
    cl::image_desc{CL_MEM_OBJECT_IMAGE2D, mask.cols(), mask.rows(),
                   0, 1, 0, 0, 0, 0, nullptr},
    cl::buffer::device);
  
  cl::image g_g(ctx,
    cl::image_format{cl::channel_order::kRGBA, cl::channel_type::kFloat},
    cl::image_desc{CL_MEM_OBJECT_IMAGE2D, mask.cols(), mask.rows(),
                   0, 1, 0, 0, 0, 0, nullptr},
    cl::buffer::device);
  
  cl::image g_guidance(ctx,
    cl::image_format{cl::channel_order::kRGBA, cl::channel_type::kFloat},
    cl::image_desc{CL_MEM_OBJECT_IMAGE2D, mask.cols(), mask.rows(),
                   0, 1, 0, 0, 0, 0, nullptr},
    cl::buffer::device);
  
  cl::write_image(g_mask,
    {0, 0, 0}, {mask.cols(), mask.rows(), 1}, mask.pitch(),
    reinterpret_cast<const uint8_t*>(mask.data()))(ctx.default_queue(), {}).wait();
  
  cl::write_image(g_f,
    {0, 0, 0}, {dst.cols(), dst.rows(), 1}, dst.pitch(),
    reinterpret_cast<const uint8_t*>(dst.data()))(ctx.default_queue(), {}).wait();
  
  cl::write_image(g_g,
    {0, 0, 0}, {src.cols(), src.rows(), 1}, src.pitch(),
    reinterpret_cast<const uint8_t*>(src.data()))(ctx.default_queue(), {}).wait();
  
  cl::invoke_kernel(make_boundary,
    {mask.cols(), mask.rows()}, std::make_tuple(g_mask, g_boundary))
    (ctx.default_queue(), {}).wait();
  
  cl::invoke_kernel(make_guidance,
    {mask.cols(), mask.rows()}, std::make_tuple(g_f, g_g, g_mask, g_boundary, g_guidance))
    (ctx.default_queue(), {}).wait();
  
  cl::invoke_kernel(apply_mask,
    {mask.cols(), mask.rows()}, std::make_tuple(g_mask, g_f))
    (ctx.default_queue(), {}).wait();
  
  for (size_t i = 0; i < 500; ++i) {
    cl::invoke_kernel(jacobi_iteration,
    {mask.cols(), mask.rows()}, std::make_tuple(g_f, g_guidance, g_mask, g_g))
    (ctx.default_queue(), {}).wait();
    g_g.swap(g_f);
  }
  
  gil::mat<uint8_t> boundary(mask.size());
  cl::read_image(g_boundary,
    {0, 0, 0}, {boundary.cols(), boundary.rows(), 1}, boundary.pitch(),
    reinterpret_cast<uint8_t*>(boundary.data()))(ctx.default_queue(), {}).wait();
  
  cv::imwrite("boundary.jpg", cv::Mat(boundary));
  
  
  gil::mat<gil::vec4f> guidance(mask.size());
  cl::read_image(g_guidance,
    {0, 0, 0}, {guidance.cols(), guidance.rows(), 1}, guidance.pitch(),
    reinterpret_cast<uint8_t*>(guidance.data()))(ctx.default_queue(), {}).wait();

  cv::imwrite("guidance.jpg", cv::Mat(guidance));
  
  gil::mat<gil::vec4f> result(mask.size());
  cl::read_image(g_f,
    {0, 0, 0}, {result.cols(), result.rows(), 1}, result.pitch(),
    reinterpret_cast<uint8_t*>(result.data()))(ctx.default_queue(), {}).wait();

  cv::imwrite("result.jpg", cv::Mat(result));
  
  /*cl::invoke_kernel(make_guidance,
    {mask.cols(), mask.rows()}, std::make_tuple(cl_mask, g_boundary))
    (ctx.default_queue(), {}).wait();*/
  
  /*gil::mat<gil::vec4b> src(gil::image_file("src.jpg").convert(gil::image_file::k32Bits));
  
  cl::image im(ctx,
    cl::image_format{cl::channel_order::kRGBx, cl::channel_type::kUNormInt8},
    cl::image_desc{CL_MEM_OBJECT_IMAGE2D, src.cols(), src.rows(), 0, 1, 0, 0, 0, 0, nullptr},
    cl::buffer::device);
  
  cl::image im2(ctx,
    cl::image_format{cl::channel_order::kRGBx, cl::channel_type::kUNormInt8},
    cl::image_desc{CL_MEM_OBJECT_IMAGE2D, src.cols(), src.rows(), 0, 1, 0, 0, 0, 0, nullptr},
    cl::buffer::device);
  cl::write_image(im,
    {0, 0, 0},
    {src.cols(), src.rows(), 1},
    src.stride() * sizeof(gil::vec4b),
    reinterpret_cast<const uint8_t*>(src.data()))(ctx.default_queue(), {});
  
  const char* source2 = R"###(
__constant sampler_t sampler =
      CLK_NORMALIZED_COORDS_FALSE
    | CLK_ADDRESS_CLAMP_TO_EDGE
    | CLK_FILTER_NEAREST;
  
__kernel void copy(__read_only image2d_t input, __write_only image2d_t output) {
  const int2 pos = {get_global_id(0), get_global_id(1)};

  float4 v = read_imagef(input, sampler, (int2)(pos.x, pos.y));
  write_imagef (output, (int2)(pos.x, pos.y), v);
}
  )###";
  
  cl::program prog2(ctx, source2);
  prog2.build();
  cl::kernel copy(prog2, "copy");
  cl::invoke_kernel(copy, {src.cols(), src.rows()}, std::make_tuple(im, im2))(ctx.default_queue(), {}).wait();
  gil::mat<gil::vec4b> dst(src.size());
  cl::read_image(im2, {0, 0, 0}, {src.cols(), src.rows(), 1}, src.stride()*sizeof(gil::vec4b), reinterpret_cast<uint8_t*>(dst.data()))(ctx.default_queue(), {});
  
  assert(gil::image_file(dst).convert(gil::image_file::k24Bits).save("bou.jpg", gil::image_file::kJpeg));
  
  cl::buffer x(ctx, 10*sizeof(float), cl::buffer::device);
  cl::buffer y(ctx, 10*sizeof(float), cl::buffer::device);
  
  cl::event e1 = cl::fill_buffer<float>(x, 0, 10, 3.0f)(ctx.default_queue(), {});
  cl::event e2 = cl::fill_buffer<float>(y, 0, 10, 5.0f)(ctx.default_queue(), {e1});
  
  const char* source = R"###(
__kernel void SAXPY (__global float* x, __global float* y, float a)
{
  const int i = get_global_id(0);

  y [i] += a * x[i];
}
  )###";
  std::cout<< source << std::endl;
  
  cl::program prog(ctx, source);
  prog.build();
  cl::kernel SAXPY(prog, "SAXPY");
  
  cl::event e3 = cl::invoke_kernel(SAXPY, {10}, std::tuple<cl::buffer, cl::buffer, float>(x, y, 3.0))(ctx.default_queue(), {e2});
  
  std::vector<float> z(10);
  cl::read_buffer(y, 0, 10, z.data())(ctx.default_queue(), {e3}).wait();
  for (auto v : z) {
    std::cout << v << std::endl;
  }*/

  /*gil::mat<gil::vec3b> src(gil::image_file("src.jpg"));
  gil::mat<gil::vec3b> dst(gil::image_file("dst.jpg"));
  gil::mat<uint8_t> mask(gil::image_file("mask.jpg", gil::image_file::kGreyScale));
  
  gil::mat<gil::vec3f> f = dst;
  gil::mat<gil::vec3f> g = src;
  
  auto start = std::chrono::high_resolution_clock::now();
  gil::mat<gil::vec3f> b = make_guidance(f, g, mask, make_boundary(mask));
  apply_mask(mask, gil::mat_view<gil::vec3f>(b));
  apply_mask(mask, gil::mat_view<gil::vec3f>(f));
  g = gil::vec3f();
  for (int i = 0; i < 200; ++i) {
    jacobi_iteration(f, b, mask, g);
    g.swap(f);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-start;
  std::cout << diff.count() << std::endl;

  copy(f, mask, dst);
  gil::image_file result(dst);
  result.save("result.jpg", gil::image_file::kJpeg);*/

  return 0;
}
