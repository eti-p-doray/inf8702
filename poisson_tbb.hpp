#pragma once

#include <assert.h>

#include "gil/mat.hpp"
#include "gil/vec.hpp"

gil::mat<uint8_t> tbb_make_boundary(gil::mat_cview<uint8_t> mask);

gil::mat<gil::vec3f> tbb_make_guidance(gil::mat_cview<gil::vec3f> f,
                                   gil::mat_cview<gil::vec3f> g,
                                   gil::mat_cview<uint8_t> mask,
                                   gil::mat_cview<uint8_t> boundary);

gil::mat<gil::vec3f> tbb_make_guidance_mixed_gradient(gil::mat_cview<gil::vec3f> f,
                                    gil::mat_cview<gil::vec3f> g,
                                    gil::mat_cview<uint8_t> mask,
                                    gil::mat_cview<uint8_t> boundary);

void tbb_jacobi_iteration(gil::mat_cview<gil::vec3f> src,
                      gil::mat_cview<gil::vec3f> b,
                      gil::mat_cview<uint8_t> mask,
                      gil::mat_view<gil::vec3f> dst);

void tbb_copy(gil::mat_cview<gil::vec3f> src,
          gil::mat_cview<uint8_t> mask,
          gil::mat_view<gil::vec3f> dst);

void tbb_copy(gil::mat_cview<gil::vec3f> src,
          gil::mat_cview<uint8_t> mask,
          gil::mat_view<gil::vec4f> dst);

template <class T>
void tbb_apply_mask(gil::mat_cview<uint8_t> mask, gil::mat_view<T> f) {
  for (int i = 0; i < f.rows(); ++i) {
    auto f_it = f.row_begin(i);
    const uint8_t* mask_it = mask.row_begin(i);
    for (size_t j = 0; j < f.cols(); ++j, ++f_it, ++mask_it) {
      if (*mask_it < 128) {
        *f_it = {};
      }
    }
  }
}
