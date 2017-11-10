#include "gil/vec.hpp"

#include "gil/mat.hpp"
#include "gil/image_file.hpp"
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

//#include <boost/gil/gil_all.hpp>

//#include "poisson.hpp"

//Jacobi method
//freeimage

// z_1 + s < a, z_2 + s < b


gil::mat<uint8_t> make_boundary(gil::mat_cview<uint8_t> mask) {
  gil::mat<uint8_t> boundary({mask.rows(), mask.cols()});
  size_t step = mask.stride();
  for (int i = 0; i < mask.rows()-1; ++i) {
    auto mask_it = mask.row_cbegin(i);
    auto bound_it = boundary.row_begin(i);
    for (size_t j = 0; j < mask.cols()-1; ++j, ++mask_it, ++bound_it) {
      if (*mask_it >= 128 && mask_it[1] < 128)
        *bound_it = 255;
      if (*mask_it < 128 && mask_it[1] >= 128)
        bound_it[1] = 255;
      if (*mask_it >= 128 && mask_it[step] < 128)
        *bound_it = 255;
      if (*mask_it < 128 && mask_it[step] >= 128)
        bound_it[step] = 255;
    }
  }
  return boundary;
}

gil::mat<gil::vec3f> make_guidance(gil::mat_cview<gil::vec3f> f, gil::mat_cview<gil::vec3f> g, gil::mat_cview<uint8_t> mask, gil::mat_cview<uint8_t> boundary) {
  assert(f.size() == mask.size());
  assert(g.size() == mask.size());
  assert(boundary.size() == mask.size());
  gil::mat<gil::vec3f> dst(f.size());
  size_t g_step = g.stride();
  size_t dst_step = dst.stride();
  for (int i = 1; i < mask.rows()-1; ++i) {
    auto mask_it = mask.row_cbegin(i)+1;
    auto bound_it = boundary.row_cbegin(i)+1;
    auto f_it = f.row_cbegin(i)+1;
    auto g_it = g.row_cbegin(i)+1;
    auto dst_it = dst.row_begin(i)+1;
    for (size_t j = 1; j < mask.cols()-1; ++j, ++mask_it, ++bound_it, ++f_it, ++g_it, ++dst_it) {
      if (*mask_it < 128) {
        *dst_it += 4.0**g_it - (g_it[-1] + g_it[1] + g_it[-g_step] + g_it[g_step]);
      }
      if (*bound_it == 255) {
        dst_it[-1] += *f_it;
        dst_it[1] += *f_it;
        dst_it[-dst_step] += *f_it;
        dst_it[dst_step] += *f_it;
      }
    }
  }
  return dst;
}

gil::mat<gil::vec3f> make_guidance2(gil::mat_cview<gil::vec3f> f, gil::mat_cview<gil::vec3f> g, gil::mat_cview<uint8_t> mask, gil::mat_cview<uint8_t> boundary) {
  assert(f.size() == mask.size());
  assert(g.size() == mask.size());
  assert(boundary.size() == mask.size());
  gil::mat<gil::vec3f> dst(f.size());
  size_t g_step = g.stride();
  size_t f_step = f.stride();
  size_t dst_step = dst.stride();
  for (int i = 1; i < mask.rows()-1; ++i) {
    auto mask_it = mask.row_cbegin(i)+1;
    auto bound_it = boundary.row_cbegin(i)+1;
    auto f_it = f.row_cbegin(i)+1;
    auto g_it = g.row_cbegin(i)+1;
    auto dst_it = dst.row_begin(i)+1;
    for (size_t j = 1; j < mask.cols()-1; ++j, ++mask_it, ++bound_it, ++f_it, ++g_it, ++dst_it) {
      if (*mask_it < 128) {
        *dst_it += std::max(*g_it - g_it[-1], *f_it - f_it[-1]);
        *dst_it += std::max(*g_it - g_it[1], *f_it - f_it[-1]);
        *dst_it += std::max(*g_it - g_it[-g_step], *f_it - f_it[-f_step]);
        *dst_it += std::max(*g_it - g_it[g_step], *f_it - f_it[f_step]);
      }
      if (*bound_it == 255) {
        dst_it[-1] += *f_it;
        dst_it[1] += *f_it;
        dst_it[-dst_step] += *f_it;
        dst_it[dst_step] += *f_it;
      }
    }
  }
  return dst;
}

template <class T>
void apply_mask(gil::mat_cview<uint8_t> mask, gil::mat_view<T> f) {
  for (int i = 0; i < f.rows(); ++i) {
    auto f_it = f.row_begin(i);
    const uint8_t* mask_it = mask.row_begin(i);
    for (size_t j = 0; j < f.cols(); ++j, ++f_it, ++mask_it) {
      if (*mask_it >= 128) {
        *f_it = {};
      }
    }
  }
}

void jacobi_iteration(gil::mat_cview<gil::vec3f> src, gil::mat_cview<gil::vec3f> b, gil::mat_cview<uint8_t> mask, gil::mat_view<gil::vec3f> dst) {
  assert(src.size() == mask.size());
  assert(b.size() == mask.size());
  assert(dst.size() == mask.size());
  size_t src_step = src.stride();
  for (int i = 1; i < src.rows()-1; ++i) {
    const gil::vec3f* src_it = src.row_cbegin(i)+1;
    const gil::vec3f* b_it = b.row_cbegin(i)+1;
    const uint8_t* mask_it = mask.row_cbegin(i)+1;
    gil::vec3f* dst_it = dst.row_begin(i)+1;
    for (size_t j = 1; j < src.cols()-1; ++j, ++src_it, ++dst_it, ++b_it, ++mask_it) {
      if (*mask_it < 128) {
        *dst_it = (*b_it + src_it[-1] + src_it[1] + src_it[-src_step] + src_it[src_step]) / 4.0;
      }
    }
  }
}

void apply_remainder(gil::mat_cview<gil::vec3f> src, gil::mat_cview<uint8_t> mask, gil::mat_view<gil::vec3f> dst) {
  size_t src_step = src.stride();
  for (int i = 1; i < src.rows()-1; ++i) {
    const gil::vec3f* src_it = src.row_cbegin(i)+1;
    const uint8_t* mask_it = mask.row_cbegin(i)+1;
    gil::vec3f* dst_it = dst.row_begin(i)+1;
    for (size_t j = 1; j < src.cols()-1; ++j, ++src_it, ++dst_it, ++mask_it) {
      if (*mask_it < 128) {
        *dst_it = (src_it[-1] + src_it[1] + src_it[-src_step] + src_it[src_step]);
      }
    }
  }
}

void copy(gil::mat_cview<gil::vec3f> src, gil::mat_cview<uint8_t> mask, gil::mat_view<gil::vec3f> dst) {
  assert(src.size() == mask.size());
  assert(dst.size() == mask.size());
  
  for (int i = 0; i < mask.rows(); ++i) {
    const uint8_t* mask_it = mask.row_cbegin(i);
    const gil::vec3f* src_it = src.row_cbegin(i);
    gil::vec3f* dst_it = dst.row_begin(i);
    for (size_t j = 0; j < mask.cols(); ++j, ++mask_it, ++src_it, ++dst_it) {
      if (*mask_it < 128) {
        *dst_it = *src_it;
      }
    }
  }
}

template <class T>
struct cv_channel {};

template <>
struct cv_channel<uint8_t> { static constexpr int value = CV_8U; };

template <>
struct cv_channel<gil::vec3b> { static constexpr int value = CV_8UC3; };

template <>
struct cv_channel<gil::vec3f> { static constexpr int value = CV_32FC3; };

template <class T>
void display(gil::mat_view<T> im, gil::vec2<size_t> size) {
  cv::namedWindow("Display window", cv::WINDOW_NORMAL); // Create a window for display.
  auto im_cv = cv::Mat(int(im.rows()), int(im.cols()), cv_channel<T>::value, im.data(), im.stride() * sizeof(T));
  cv::Mat tmp;
  cv::resize(im_cv, tmp, cv::Size(int(size[1]), int(size[0])));
  cv::imshow("Display window", tmp);  // Show our image inside it.
  cv::resizeWindow("Display window", int(size[1]), int(size[0]));

  cv::waitKey(0); // Wait for a keystroke in the window
}

int main() {
  using namespace std::placeholders;
  
  std::cout << (gil::vec3f{2.0, 0.0, 0.0} < gil::vec3f{1.0, 3.0, 0.0}) <<std::endl;

  gil::mat<gil::vec3b> water(image_file("water.jpg"));
  gil::mat<gil::vec3b> dog(image_file("dog.jpg"));
  gil::mat<uint8_t> mask(image_file("mask.jpg", image_file::kGreyScale));
  
  gil::mat<gil::vec3f> f = water[gil::vec4<size_t>{200, 200, 600, 600}];
  gil::mat<gil::vec3f> g = dog[gil::vec4<size_t>{300, 600, 600, 600}];
  
  gil::mat<gil::vec3f> b = make_guidance2(f, g, mask, make_boundary(mask));
  apply_mask(mask, gil::mat_view<gil::vec3f>(b));
  apply_mask(mask, gil::mat_view<gil::vec3f>(g));
  f = gil::vec3f();
  for (int i = 0; i < 200; ++i) {
    jacobi_iteration(g, b, mask, f);
    g.swap(f);
  }

  f = water[gil::vec4<size_t>{200, 200, 600, 600}];
  copy(g, mask, f);
  //cat[gil::vec4<size_t>{200, 2000, 600, 600}] = gil::mat_view<gil::vec3f>(f);
  
  /*gil::mat<gil::vec3f> tmp(g.size());
  apply_mask(mask, gil::mat_view<gil::vec3f>(g));
  apply_remainder(g, mask, tmp);
  g.apply(std::bind(acier::multiplies(), _1, 4.0));
  g -= tmp;
  g -= b;*/
  
  //f -= b;

  f.apply(std::bind(acier::divides(), _1, 255.0));
  display<gil::vec3f>(f, {600, 600});
  

  //display(mask);
  return 0;
}
