
#include <chrono>
#include <iostream>

#include "gil/mat.hpp"
#include "gil/vec.hpp"
#include "gil/image_file.hpp"
#include "poisson.hpp"

int main() {
  using namespace std::placeholders;

  gil::mat<gil::vec3b> src(gil::image_file("src.jpg"));
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
  result.save("result.jpg", gil::image_file::kJpeg);

  return 0;
}
