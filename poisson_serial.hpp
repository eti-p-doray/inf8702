#pragma

#include <assert.h>

#include "gil/mat.hpp"
#include "gil/vec.hpp"

gil::mat<uint8_t> make_boundary(gil::mat_cview<uint8_t> mask);

gil::mat<gil::vec3f> make_guidance(gil::mat_cview<gil::vec3f> f,
                                   gil::mat_cview<gil::vec3f> g,
                                   gil::mat_cview<uint8_t> mask,
                                   gil::mat_cview<uint8_t> boundary);

gil::mat<gil::vec3f> make_guidance_mixed_gradient(gil::mat_cview<gil::vec3f> f,
                                    gil::mat_cview<gil::vec3f> g,
                                    gil::mat_cview<uint8_t> mask,
                                    gil::mat_cview<uint8_t> boundary);

gil::mat<gil::vec3f> make_guidance_mixed_gradient_avg(gil::mat_cview<gil::vec3f> f,
                                    gil::mat_cview<gil::vec3f> g,
                                    gil::mat_cview<uint8_t> mask,
                                    gil::mat_cview<uint8_t> boundary);

void jacobi_iteration(gil::mat_cview<gil::vec3f> src,
                      gil::mat_cview<gil::vec3f> b,
                      gil::mat_cview<uint8_t> mask,
                      gil::mat_view<gil::vec3f> dst);

void apply_remainder(gil::mat_cview<gil::vec3f> src,
                     gil::mat_cview<uint8_t> mask,
                     gil::mat_view<gil::vec3f> dst);

void copy(gil::mat_cview<gil::vec3f> src,
          gil::mat_cview<uint8_t> mask,
          gil::mat_view<gil::vec3f> dst);

void copy(gil::mat_cview<gil::vec3f> src,
          gil::mat_cview<uint8_t> mask,
          gil::mat_view<gil::vec4f> dst);

template <class T>
void apply_mask(gil::mat_cview<uint8_t> mask, gil::mat_view<T> f) {
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
