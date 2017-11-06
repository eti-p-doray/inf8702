#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

//Jacobi method

// z_1 + s < a, z_2 + s < b

cv::Mat make_boundary(const cv::Mat& mask) {
  cv::Mat boundary(mask.rows, mask.cols, mask.depth(), cv::Scalar(0));
  size_t step = mask.step1();
  for (size_t i = 0; i < mask.rows-1; ++i) {
    const uint8_t* mask_it = mask.template ptr<uint8_t>(i);
    uint8_t* bound_it = boundary.template ptr<uint8_t>(i);
    for (size_t j = 0; j < mask.cols-1; ++j, ++mask_it, ++bound_it) {
      if (bool(*mask_it > 128) != bool(mask_it[1] > 128)) {
        *bound_it = 255;
        bound_it[1] = 255;
      }
      if (bool(*mask_it > 128) != bool(mask_it[step] > 128)) {
        *bound_it = 255;
        bound_it[step] = 255;
      }
    }
  }
  return boundary;
}

cv::Mat make_target(const cv::Mat& f, const cv::Mat& g, const cv::Mat& mask, const cv::Mat& boundary) {
  assert(f.size == mask.size);
  assert(g.size == mask.size);
  assert(boundary.size == mask.size);
  cv::Mat dst(f.rows, f.cols, f.depth(), cv::Scalar(0));
  size_t step = dst.step1();
  for (size_t i = 1; i < mask.rows-1; ++i) {
    const uint8_t* mask_it = mask.template ptr<uint8_t>(i);
    const uint8_t* bound_it = boundary.template ptr<uint8_t>(i);
    const cv::Vec3b* f_it = f.template ptr<cv::Vec3b>(i);
    const cv::Vec3b* g_it = g.template ptr<cv::Vec3b>(i);
    cv::Vec3b* dst_it = dst.template ptr<cv::Vec3b>(i);
    for (size_t j = 1; j < mask.cols-1; ++j, ++mask_it, ++bound_it, ++f_it, ++g_it, ++dst_it) {
      auto value = *g_it;
      if (*bound_it == 0) {
        value += *f_it;
      }
      dst_it[-step] += value;
      dst_it[step] += value;
      dst_it[-1] += value;
      dst_it[1] += value;
      *dst_it -= 4*(*g_it);
    }
  }
  return dst;
}

void copy(const cv::Mat& src, const cv::Mat& mask, cv::Mat dst) {
  assert(src.size == mask.size);
  assert(dst.size == mask.size);
  assert(mask.depth() == CV_8U);
  
  for (size_t i = 0; i < mask.rows; ++i) {
    const uint8_t* mask_it = mask.template ptr<uint8_t>(i);
    const cv::Vec3b* src_it = src.template ptr<cv::Vec3b>(i);
    cv::Vec3b* dst_it = dst.template ptr<cv::Vec3b>(i);
    for (size_t j = 0; j < mask.cols; ++j, ++mask_it, ++src_it, ++dst_it) {
      if (*mask_it == 0) {
        *dst_it = *src_it;
      }
    }
  }
  
  //src.copyTo(dst);
}

int main() {
  cv::Mat cat = cv::imread("cat.jpg", CV_LOAD_IMAGE_COLOR); // Read the file
  cv::Mat dog = cv::imread("dog.jpg", CV_LOAD_IMAGE_COLOR); // Read the file
  cv::Mat mask = cv::imread("mask.jpg", CV_LOAD_IMAGE_COLOR); // Read the file
  
  cv::cvtColor(mask, mask, cv::COLOR_RGB2GRAY);
  
  //cv::Mat mask =
  std::cout << cat.size[0] << " " << cat.size[1] << std::endl;
  std::cout << dog.size[0] << " " << dog.size[1] << std::endl;
  std::cout << mask.size[0] << " " << mask.size[1] << std::endl;

  //dog(cv::Rect(200, 50, 900, 500)).copyTo(cat(cv::Rect(200, 500, 900, 500)));
  
  copy(dog(cv::Rect(200, 0, 600, 600)), mask, cat(cv::Rect(200, 400, 600, 600)));
  auto f = cat(cv::Rect(200, 400, 600, 600));
  auto g = dog(cv::Rect(200, 0, 600, 600));

  cv::namedWindow("Display window", cv::WINDOW_NORMAL); // Create a window for display.
  cv::resize(cat, cat, cv::Size(613, 400));
  cv::imshow("Display window", make_target(f, g, mask, make_boundary(mask)));  // Show our image inside it.
  cv::resizeWindow("Display window", 613, 400);

  cv::waitKey(0); // Wait for a keystroke in the window
  return 0;
}
