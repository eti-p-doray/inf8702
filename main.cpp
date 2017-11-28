
#include <chrono>
#include <iostream>

#include <boost/filesystem.hpp>
#include <boost/iostreams/device/mapped_file.hpp>

#include <opencv2/highgui/highgui.hpp>

#include "gil/mat.hpp"
#include "gil/vec.hpp"
#include "poisson_serial.hpp"
#include "poisson_tbb.hpp"

#include "cl/device.hpp"
#include "cl/context.hpp"
#include "cl/memory.hpp"
#include "cl/program.hpp"
#include "cl/kernel.hpp"

/**
 * Finds the full path of |filename| in the working directory.
 */
boost::filesystem::path find_file(const std::string& filename) {
  boost::filesystem::path path(__FILE__);
  path.remove_filename();
  path.append(filename);
  return path;
}

/**
 * Finds the smallest possible rectangular frame that contains the |mask|'s
 * white region, in other words the minimal region we'll need to consider to
 * apply poisson blending. Returns a vector with the 4 extremums of the frame.
 */
gil::vec4<size_t> find_frame(gil::mat_cview<uint8_t> mask) {
  gil::vec4<size_t> frame = {-1, -1, 0, 0};
  for (size_t i = 0; i < mask.rows(); ++i) {
    auto elem_it = mask.row_begin(i);
    for (size_t j = 0; j < mask.cols(); ++j, ++elem_it) {
      // If the pixel is white, hence should be in the frame
      if (*elem_it >= 128) {
        // If the pixel is to the left of the frame, set it as the new leftmost
        if (i <= frame[0])
          frame[0] = i - 1;

        // If the pixel is atop the frame, set it as the new topmost
        if (j <= frame[1])
          frame[1] = j - 1;

        // If the pixel is to the right of the frame, set it as the new rightmost
        if (i >= frame[2])
          frame[2] = i + 1;

        // If the pixel is under the frame, set it as the new bottommost
        if (j >= frame[3])
          frame[3] = j + 1;
      }
    }
  }
  frame[0] -= 1;
  frame[1] -= 1;
  frame[2] -= frame[0] - 2; // adjust indexes relative to the min, with a 1px border
  frame[3] -= frame[1] - 2;
  return frame;
}

/**
 * Applies poisson blending on a single process. Finds a patch by applying |mask|
 * upon |src| and blend this patch on |dst| at the corresponding region (again
 * described by applying |mask|). The result of the blending is put in the
 * output parameter |result|.
 */
 void poisson_blending_serial(gil::mat_cview<uint8_t> mask,
                      gil::mat_cview<gil::vec3f> src,
                      gil::mat_cview<gil::vec3f> dst,
                      gil::mat_view<gil::vec3f> result) {

  assert(src.size() == mask.size());
  assert(dst.size() == mask.size());

  // Formula applied here : for all p in the destination domain (omega)
  // |N_p| * f_p - sum[all q in (N_p intersection omega)]{f_q} =
  // sum[all q in (N_p intersection delta_omega)]{f*_q} + sum[all q in N_p]{v_pq}
  // (equation 7 of http://www.cs.virginia.edu/~connelly/class/2014/comp_photo/proj2/poisson.pdf)
  // Where N_p are the neighbooring 4 pixels to p, f_p is the intensity of the source at p,
  // delta_omega is the boundary's domain, f*_q the intensity of the destination at q
  // and v_pq is the vector guidance field's value for the point between p and q,
  // ie. v_pq = g_p - g_q, with g_{something} being the source image's value at "something"
  // Do note that we do not reuse this notation.

  // calculate the right side of the equation, see above. Constant across solving
  auto b = make_guidance_mixed_gradient(dst, src, mask, make_boundary(mask));
  apply_mask(mask, b); // select the part corresponding to the mask's region
  gil::mat<gil::vec3f> f(dst.size()); //Will contain the intensity of the image used as input to get f_p
  gil::mat<gil::vec3f> g(dst.size()); //Will contain the output of one iteration
  copy(dst, mask, f); //applies the mask on destination and put the output as a copy in f.
  for (int i = 0; i < 500; ++i) { // applying iterative method to have the value of f converge
    jacobi_iteration(f, b, mask, g); // Calculate the new value of g
    f.swap(g); // use g as an input for next iteration
  }
  result = dst;
  copy(f, mask, result); // put resulting f's area corresponding to mask in
                         // output |result| variable to return
}

/**
 * Applies poisson blending on a multiple cpu process. Finds a patch by applying
 * |mask|upon |src| and blend this patch on |dst| at the corresponding region
 * (again described by applying |mask|). The result of the blending is put in
 * the output parameter |result|.
 */
void poisson_blending_tbb(gil::mat_cview<uint8_t> mask,
                             gil::mat_cview<gil::vec3f> src,
                             gil::mat_cview<gil::vec3f> dst,
                             gil::mat_view<gil::vec3f> result) {

  assert(src.size() == mask.size());
  assert(dst.size() == mask.size());

  // Formula applied here : for all p in the destination domain (omega)
  // |N_p| * f_p - sum[all q in (N_p intersection omega)]{f_q} =
  // sum[all q in (N_p intersection delta_omega)]{f*_q} + sum[all q in N_p]{v_pq}
  // (equation 7 of http://www.cs.virginia.edu/~connelly/class/2014/comp_photo/proj2/poisson.pdf)
  // Where N_p are the neighbooring 4 pixels to p, f_p is the intensity of the source at p,
  // delta_omega is the boundary's domain, f*_q the intensity of the destination at q
  // and v_pq is the vector guidance field's value for the point between p and q,
  // ie. v_pq = g_p - g_q, with g_{something} being the source image's value at "something"
  // Do note that we do not reuse this notation.

  // calculate the right side of the equation, see above. Constant across solving
  auto b = tbb_make_guidance_mixed_gradient(dst, src, mask, tbb_make_boundary(mask));
  tbb_apply_mask(mask, b); // select the part corresponding to the mask's region
  gil::mat<gil::vec3f> f(dst.size()); //Will contain the intensity of the image used as input to get f_p
  gil::mat<gil::vec3f> g(dst.size()); //Will contain the output of one iteration
  tbb_copy(dst, mask, f); //applies the mask on destination and put the output as a copy in f.
  for (int i = 0; i < 500; ++i) { // applying iterative method to have the value of f converge
    tbb_jacobi_iteration(f, b, mask, g); // Calculate the new value of g
    f.swap(g); // use g as an input for next iteration
  }
  result = dst;
  tbb_copy(f, mask, result); // put resulting f's area corresponding to mask in
  // output |result| variable to return
}

// Class to be used to execute the poisson blending with OpenCL.
// In a class to compile the OpenCL program on c++ compilation
class poisson_blending_cl {
 public:
  // Constructor, builds the OpenCL program
  poisson_blending_cl()
      : device_(cl::get_devices(cl::filter::gpu())[0]),
        ctx_(device_) {
    // read and load the OpenCL program from its file
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

  /**
   * Operator calculating the poisson blending using the OpenCL program.
   * Finds a patch by applying |mask| upon |src| and blend this patch on |dst| at
   * the corresponding region (again described by applying |mask|). The result
   * of the blending is put in the output parameter |result|.
   */
  void operator()(gil::mat_cview<uint8_t> mask,
                         gil::mat_cview<gil::vec3f> src,
                         gil::mat_cview<gil::vec3f> dst,
                         gil::mat_view<gil::vec3f> result) {

    // Formula applied here : for all p in the destination domain (omega)
    // |N_p| * f_p - sum[all q in (N_p intersection omega)]{f_q} =
    // sum[all q in (N_p intersection delta_omega)]{f*_q} + sum[all q in N_p]{v_pq}
    // (equation 7 of http://www.cs.virginia.edu/~connelly/class/2014/comp_photo/proj2/poisson.pdf)
    // Where N_p are the neighbooring 4 pixels to p, f_p is the intensity of the source at p,
    // delta_omega is the boundary's domain, f*_q the intensity of the destination at q
    // and v_pq is the vector guidance field's value for the point between p and q,
    // ie. v_pq = g_p - g_q, with g_{something} being the source image's value at "something"
    // Do note that we do not reuse this notation.

    // OpenCl image of the mask
    cl::image cl_mask(ctx_,
      cl::image_format{cl::channel_order::kR, cl::channel_type::kUInt8},
      cl::image_desc::make_image_2d(mask.cols(), mask.rows()),
      cl::buffer::device);

    // OpenCl image of the boundary, used to calculate the right side of the poisson equation
    cl::image cl_boundary(ctx_,
      cl::image_format{cl::channel_order::kR, cl::channel_type::kUInt8},
      cl::image_desc::make_image_2d(mask.cols(), mask.rows()),
      cl::buffer::device);

    // OpenCl image to contain the intensity values from last iteration
    cl::image cl_f(ctx_,
      cl::image_format{cl::channel_order::kRGB, cl::channel_type::kFloat},
      cl::image_desc::make_image_2d(mask.cols(), mask.rows()),
      cl::buffer::device);

    // OpenCl image to contain the result of poisson iterations
    cl::image cl_g(ctx_,
      cl::image_format{cl::channel_order::kRGB, cl::channel_type::kFloat},
      cl::image_desc::make_image_2d(mask.cols(), mask.rows()),
      cl::buffer::device);

    // OpenCl image to contain the right side of the equation
    cl::image cl_guidance(ctx_,
      cl::image_format{cl::channel_order::kRGB, cl::channel_type::kFloat},
      cl::image_desc::make_image_2d(mask.cols(), mask.rows()),
      cl::buffer::device);

    // Initialise cl_mask using the mask image data
    cl::write_image(cl_mask,
      {0, 0, 0}, {mask.cols(), mask.rows(), 1}, mask.pitch(),
      reinterpret_cast<const uint8_t*>(mask.data()))
      (ctx_.default_queue(), {}).wait();

    // Initialise cl_f using the destination image data
    cl::write_image(cl_f,
      {0, 0, 0}, {dst.cols(), dst.rows(), 1}, dst.pitch(),
      reinterpret_cast<const uint8_t*>(dst.data()))
      (ctx_.default_queue(), {}).wait();

    // Initialise cl_g using the source image data
    cl::write_image(cl_g,
      {0, 0, 0}, {src.cols(), src.rows(), 1}, src.pitch(),
      reinterpret_cast<const uint8_t*>(src.data()))
      (ctx_.default_queue(), {}).wait();

    // Initialise cl_boundary by calculating the cl_mask's boundary
    cl::invoke_kernel(make_boundary_,
      {mask.cols(), mask.rows()}, std::make_tuple(cl_mask, cl_boundary))
      (ctx_.default_queue(), {}).wait();

    // Initialise cl_guidance by calculating the right side of the poisson equation.
    // We save the event of that call's end in e1.
    auto e1 = cl::invoke_kernel(make_guidance_,
      {mask.cols(), mask.rows()},
      std::make_tuple(cl_f, cl_g, cl_mask, cl_boundary, cl_guidance))
      (ctx_.default_queue(), {});

    // We apply the cl_mask on cl_f in order to select only the relevant information
    // from the destination
    cl::invoke_kernel(apply_mask_,
      {mask.cols(), mask.rows()}, std::make_tuple(cl_mask, cl_f))
      (ctx_.default_queue(), {}).wait();

    // Using iterative method to calculate cl_g
    for (size_t i = 0; i < 500; ++i) {
      // Once e1 happened (guidance field complete for first iteration,
      // previous iteration for the 499 other iterations), calculate
      // a new value of intensity field based on the left side of the equation
      e1 = cl::invoke_kernel(jacobi_iteration_,
      {mask.cols(), mask.rows()},
      std::make_tuple(cl_f, cl_guidance, cl_mask, cl_g))
      (ctx_.default_queue(), {e1});
      cl_g.swap(cl_f);
    }

    result = dst;
    gil::mat<gil::vec3f> tmp(dst.size());
    // once the last iteration is done, copy the resulting cl_f into tmp matrix.
    cl::read_image(cl_f,
      {0, 0, 0}, {tmp.cols(), tmp.rows(), 1}, tmp.pitch(),
      reinterpret_cast<uint8_t*>(tmp.data()))(ctx_.default_queue(), {e1}).wait();
    copy(tmp, mask, result); // Apply mask on tmp and paste the output at the corresponding
                             // region onto result, initialised with destination
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

  // Load images
  gil::mat<uint8_t> mask(gil::mat_view<gil::vec3b>(cv::imread("mask.jpg")));
  gil::mat<gil::vec3f> src(gil::mat_view<gil::vec3b>(
      cv::imread("src.jpg", CV_LOAD_IMAGE_COLOR)));
  gil::mat<gil::vec3f> dst(gil::mat_view<gil::vec3b>(
      cv::imread("dst.jpg", CV_LOAD_IMAGE_COLOR)));
  gil::mat<gil::vec3f> result = dst;
  auto frame = find_frame(mask); // limit the mask's size to the minimum needed

  // Time the serial calculation of serial poisson blending and save its output in a file
  auto start = std::chrono::high_resolution_clock::now();
  poisson_blending_serial(mask[frame], src[frame], dst[frame], result[frame]);
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end-start;
  std::cout << diff.count() << std::endl;
  cv::imwrite("result1.jpg", cv::Mat(result));

  // Time the opencl calculation of serial poisson blending and save its output in a file
  poisson_blending_cl poisson_blending_cl;
  start = std::chrono::high_resolution_clock::now();
  poisson_blending_cl(mask[frame], src[frame], dst[frame], result[frame]);
  end = std::chrono::high_resolution_clock::now();
  diff = end-start;
  std::cout << diff.count() << std::endl;
  cv::imwrite("result2.jpg", cv::Mat(result));

  // Time the tbb calculation of serial poisson blending and save its output in a file
  start = std::chrono::high_resolution_clock::now();
  poisson_blending_tbb(mask[frame], src[frame], dst[frame], result[frame]);
  end = std::chrono::high_resolution_clock::now();
  diff = end-start;
  std::cout << diff.count() << std::endl;
  cv::imwrite("result3.jpg", cv::Mat(result));

  return 0;
}
