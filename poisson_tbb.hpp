#pragma once

#include <assert.h>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

#include "gil/mat.hpp"
#include "gil/vec.hpp"

#include "poisson.hpp"

gil::mat<uint8_t> tbb_make_boundary(gil::mat_cview<uint8_t> mask);

gil::mat<gil::vec3f> tbb_make_guidance(gil::mat_cview<gil::vec3f> f,
                                   gil::mat_cview<gil::vec3f> g,
                                   gil::mat_cview<uint8_t> mask,
                                   GradientMethod method);

gil::mat<gil::vec3f> tbb_make_guidance(gil::mat_cview<gil::vec3f> f,
                                   gil::mat_cview<gil::vec3f> g,
                                   gil::mat_cview<uint8_t> mask,
                                   gil::mat_cview<uint8_t> boundary);

gil::mat<gil::vec3f> tbb_make_guidance_mixed_gradient(gil::mat_cview<gil::vec3f> f,
                                    gil::mat_cview<gil::vec3f> g,
                                    gil::mat_cview<uint8_t> mask,
                                    gil::mat_cview<uint8_t> boundary);


gil::mat<gil::vec3f> tbb_make_guidance_mixed_gradient_avg(gil::mat_cview<gil::vec3f> f,
                                    gil::mat_cview<gil::vec3f> g,
                                    gil::mat_cview<uint8_t> mask,
                                    gil::mat_cview<uint8_t> boundary);

void tbb_jacobi_iteration(gil::mat_cview<gil::vec3f> src,
                      gil::mat_cview<gil::vec3f> b,
                      gil::mat_cview<uint8_t> mask,
                      gil::mat_view<gil::vec3f> dst);


/**
 * Class used by tbb to apply the parallel_for applying the mask
 */
template <class T>
class ParallelApplyMask {
public:
  ParallelApplyMask(const gil::mat_cview<uint8_t> mask, gil::mat_view<T> image)
    : mask_(mask),  image_(image) {
    //empty, all in initialisation list
  }

  void operator()(const tbb::blocked_range<size_t>& range) const {
    for (size_t i = range.begin(); i != range.end(); ++i) {
      auto f_it = image_.row_begin(i);
      const uint8_t* mask_it = mask_.row_begin(i);
      for (size_t j = 0; j < image_.cols(); ++j, ++f_it, ++mask_it) {
        if (*mask_it < 128) {
          *f_it = {};
        }
      }
    }
  }

private:
  gil::mat_cview<uint8_t> mask_;
  gil::mat_view<T> image_;
};

template <class T>
void tbb_apply_mask(gil::mat_cview<uint8_t> mask, gil::mat_view<T> f) {
  ParallelApplyMask<T> para_apply_mask(mask, f);
  tbb::parallel_for(tbb::blocked_range<size_t>(0, f.rows()), para_apply_mask);
}
