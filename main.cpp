
#include <chrono>
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include <opencv2/highgui/highgui.hpp>

#include "gil/mat.hpp"
#include "gil/vec.hpp"
#include "poisson_serial.hpp"

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

void poisson_blending_serial(gil::mat_cview<uint8_t> mask,
                      gil::mat_cview<gil::vec3f> src,
                      gil::mat_cview<gil::vec3f> dst,
                      gil::mat_view<gil::vec3f> result) {
  
  
  auto b = make_guidance(dst, src, mask, make_boundary(mask));
  apply_mask(mask, b);
  gil::mat<gil::vec3f> f(dst.size());
  gil::mat<gil::vec3f> g(src.size());
  copy(dst, mask, g);
  for (int i = 0; i < 500; ++i) {
    jacobi_iteration(g, b, mask, f);
    g.swap(f);
  }
  result = dst;
  copy(g, mask, result);
}

class poisson_blending_cl {
 public:
  poisson_blending_cl()
      : device_(cl::get_devices(cl::filter::gpu())[0]),
        ctx_(device_) {
    boost::iostreams::mapped_file_source poisson_source(find_file("poisson.cl"));
    program_ = cl::program(ctx_, poisson_source.data());
    try {
      program_.build();
    } catch (...) {
      std::cout << program_.get_build_info<cl::program::BuildLog>(device_);
      return;
    }
    
    make_boundary_ = cl::kernel(program_, "make_boundary");
    make_guidance_ = cl::kernel(program_, "make_guidance");
    jacobi_iteration_ = cl::kernel(program_, "jacobi_iteration");
    apply_mask_ = cl::kernel(program_, "apply_mask");
  }
  
  void operator()(gil::mat_cview<uint8_t> mask,
                         gil::mat_cview<gil::vec3f> src,
                         gil::mat_cview<gil::vec3f> dst,
                         gil::mat_view<gil::vec3f> result) {
    cl::image cl_mask(ctx_,
      cl::image_format{cl::channel_order::kR, cl::channel_type::kUInt8},
      cl::image_desc::make_image_2d(mask.cols(), mask.rows()),
      cl::buffer::device);
    
    cl::image cl_boundary(ctx_,
      cl::image_format{cl::channel_order::kR, cl::channel_type::kUInt8},
      cl::image_desc::make_image_2d(mask.cols(), mask.rows()),
      cl::buffer::device);
    
    cl::image cl_f(ctx_,
      cl::image_format{cl::channel_order::kRGB, cl::channel_type::kFloat},
      cl::image_desc::make_image_2d(mask.cols(), mask.rows()),
      cl::buffer::device);
    
    cl::image cl_g(ctx_,
      cl::image_format{cl::channel_order::kRGB, cl::channel_type::kFloat},
      cl::image_desc::make_image_2d(mask.cols(), mask.rows()),
      cl::buffer::device);
    
    cl::image cl_guidance(ctx_,
      cl::image_format{cl::channel_order::kRGB, cl::channel_type::kFloat},
      cl::image_desc::make_image_2d(mask.cols(), mask.rows()),
      cl::buffer::device);
    
    cl::write_image(cl_mask,
      {0, 0, 0}, {mask.cols(), mask.rows(), 1}, mask.pitch(),
      reinterpret_cast<const uint8_t*>(mask.data()))
      (ctx_.default_queue(), {}).wait();
    
    cl::write_image(cl_f,
      {0, 0, 0}, {dst.cols(), dst.rows(), 1}, dst.pitch(),
      reinterpret_cast<const uint8_t*>(dst.data()))
      (ctx_.default_queue(), {}).wait();
    
    cl::write_image(cl_g,
      {0, 0, 0}, {src.cols(), src.rows(), 1}, src.pitch(),
      reinterpret_cast<const uint8_t*>(src.data()))
      (ctx_.default_queue(), {}).wait();
    
    cl::invoke_kernel(make_boundary_,
      {mask.cols(), mask.rows()}, std::make_tuple(cl_mask, cl_boundary))
      (ctx_.default_queue(), {}).wait();
    
    auto e1 = cl::invoke_kernel(make_guidance_,
      {mask.cols(), mask.rows()},
      std::make_tuple(cl_f, cl_g, cl_mask, cl_boundary, cl_guidance))
      (ctx_.default_queue(), {});
    
    cl::invoke_kernel(apply_mask_,
      {mask.cols(), mask.rows()}, std::make_tuple(cl_mask, cl_f))
      (ctx_.default_queue(), {}).wait();

    for (size_t i = 0; i < 500; ++i) {
      e1 = cl::invoke_kernel(jacobi_iteration_,
      {mask.cols(), mask.rows()},
      std::make_tuple(cl_f, cl_guidance, cl_mask, cl_g))
      (ctx_.default_queue(), {e1});
      cl_g.swap(cl_f);
    }
    
    result = dst;
    cl::read_image(cl_f,
      {0, 0, 0}, {result.cols(), result.rows(), 1}, result.pitch(),
      reinterpret_cast<uint8_t*>(result.data()))(ctx_.default_queue(), {e1}).wait();
  }
  
 private:
  cl::device device_;
  cl::context ctx_;
  cl::program program_;
  cl::kernel make_boundary_;
  cl::kernel make_guidance_;
  cl::kernel jacobi_iteration_;
  cl::kernel apply_mask_;
};

int main() {
  using namespace std::placeholders;
  
  gil::mat<uint8_t> mask(gil::mat_view<gil::vec3b>(cv::imread("mask.jpg")));
  gil::mat<gil::vec3f> src(gil::mat_view<gil::vec3b>(
      cv::imread("src.jpg", CV_LOAD_IMAGE_COLOR)));
  gil::mat<gil::vec3f> dst(gil::mat_view<gil::vec3b>(
      cv::imread("dst.jpg", CV_LOAD_IMAGE_COLOR)));
  gil::mat<gil::vec3f> result(dst.size());
  
  poisson_blending_cl poisson_blending_cl;
  
  auto start = std::chrono::high_resolution_clock::now();
  poisson_blending_serial(mask, src, dst, result);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-start;
  std::cout << diff.count() << std::endl;
  cv::imwrite("result1.jpg", cv::Mat(result));
  
  start = std::chrono::high_resolution_clock::now();
  poisson_blending_cl(mask, src, dst, result);
  end = std::chrono::high_resolution_clock::now();
  diff = end-start;
  std::cout << diff.count() << std::endl;
  cv::imwrite("result2.jpg", cv::Mat(result));

  return 0;
}
