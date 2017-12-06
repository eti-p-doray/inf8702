#include "poisson_serial.hpp"

#include <algorithm>
#include <cmath>

/**
 * Calculates the boundary delta_omega of the region delimited by |mask|
 */
gil::mat<uint8_t> make_boundary(gil::mat_cview<uint8_t> mask) {
  gil::mat<uint8_t> boundary({mask.rows(), mask.cols()});
  size_t mask_step = mask.stride();
  size_t bound_step = boundary.stride();
  for (int i = 0; i < mask.rows()-1; ++i) {
    auto mask_it = mask.row_cbegin(i);
    auto bound_it = boundary.row_begin(i);
    for (size_t j = 0; j < mask.cols()-1; ++j, ++mask_it, ++bound_it) {
      //if next pixel (x) is part of the mask and we're not currently in the mask
      if (*mask_it < 128 && mask_it[1] >= 128)
        *bound_it = 255; // add current pixel to boundary
      //if next pixel (x) is not part of the mask and we're currently in the mask
      if (*mask_it >= 128 && mask_it[1] < 128)
        bound_it[1] = 255; // add next pixel (x) to boundary
      //if next pixel (y) is part of the mask and we're not currently in the mask
      if (*mask_it < 128 && mask_it[mask_step] >= 128)
        *bound_it = 255; // add current pixel to boundary
      //if next pixel (x) is not part of the mask and we're currently in the mask
      if (*mask_it >= 128 && mask_it[mask_step] < 128)
        bound_it[bound_step] = 255; // add next pixel (y) to boundary
    }
  }
  return boundary;
}

gil::mat<gil::vec3f> make_guidance(gil::mat_cview<gil::vec3f> f,
                                   gil::mat_cview<gil::vec3f> g,
                                   gil::mat_cview<uint8_t> mask,
                                   GradientMethod method) {
  // calculate the right side of the equation, see above. Constant across solving
  switch (method) {
    default:
    case GradientMethod::BASE:
      return make_guidance(f, g, mask, make_boundary(mask));

    case GradientMethod::MAX_MIXING:
      return make_guidance_mixed_gradient(f, g, mask, make_boundary(mask));

    case GradientMethod::AVG_MIXING:
      return make_guidance_mixed_gradient_avg(f, g, mask, make_boundary(mask));
  }
}

/**
 * Calculates the guidance field composed of the |boundary| in destination image |f|
 * and the vector field corresponding to the |mask|'s area in |g|
 */
gil::mat<gil::vec3f> make_guidance(gil::mat_cview<gil::vec3f> f,
                                   gil::mat_cview<gil::vec3f> g,
                                   gil::mat_cview<uint8_t> mask,
                                   gil::mat_cview<uint8_t> boundary) {
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
      if (*mask_it >= 128) { // if the current pixel is in the mask
        //2nd summation of the right side of the equation, we calculate the sum of all 4 vectors
        *dst_it += 4.0**g_it - (g_it[-1] + g_it[1] + g_it[-g_step] + g_it[g_step]);
      }
      if (*bound_it == 255) { // if the pixel is on the boundary
        // 1st part of the right side of the equation, in the form of adding f* to
        // the 4 neighboors of a boundary pixel
        dst_it[-1] += *f_it;
        dst_it[1] += *f_it;
        dst_it[-dst_step] += *f_it;
        dst_it[dst_step] += *f_it;
      }
    }
  }
  return dst;
}

/**
 * Calculates the guidance field composed of the |boundary| in destination image |f|
 * and the vector field corresponding to the |mask|'s area in |g|.
 * This implementation uses mixed_gradients instead of g_p - g_q, which means that
 * we pick the max between the gradient in source and in destination
 */
gil::mat<gil::vec3f> make_guidance_mixed_gradient(gil::mat_cview<gil::vec3f> f,
                                    gil::mat_cview<gil::vec3f> g,
                                    gil::mat_cview<uint8_t> mask,
                                    gil::mat_cview<uint8_t> boundary) {
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
      if (*mask_it >= 128) { // if current pixel is part of the mask
        //vector containing the 2 potential values of v for each 4 neighboors
        // First dimension tells if f* or g is used, second to tell which neighboor
        gil::vec3f v[2][4] = {{*g_it - g_it[-1],
                         *g_it - g_it[1],
                         *g_it - g_it[-g_step],
                         *g_it - g_it[g_step]},
                        {*f_it - f_it[-1],
                         *f_it - f_it[1],
                         *f_it - f_it[-f_step],
                         *f_it - f_it[f_step]}};
        for (int k = 0; k < 4; ++k) { // for each neighboor
          //Use the highest gradient between the one from f* and the one from g
          int x = gil::norm2(v[0][k]) > gil::norm2(v[1][k]) ? 0 : 1;
          (*dst_it) += v[x][k];
        }
      }
      if (*bound_it == 255) { // if current pixel is part of the boundary
        // 1st part of the right side of the equation, in the form of adding f* to
        // the 4 neighboors of a boundary pixel
        dst_it[-1] += *f_it;
        dst_it[1] += *f_it;
        dst_it[-dst_step] += *f_it;
        dst_it[dst_step] += *f_it;
      }
    }
  }
  return dst;
}

/**
 * Calculates the guidance field composed of the |boundary| in destination image |f|
 * and the vector field corresponding to the |mask|'s area in |g|.
 * This implementation uses mixed_gradients with average instead of g_p - g_q,
 * which means that we pick the average between the gradient in source and in
 * destination.
 */
gil::mat<gil::vec3f> make_guidance_mixed_gradient_avg(gil::mat_cview<gil::vec3f> f,
                                    gil::mat_cview<gil::vec3f> g,
                                    gil::mat_cview<uint8_t> mask,
                                    gil::mat_cview<uint8_t> boundary) {
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
      if (*mask_it >= 128) { // if current pixel is part of the mask
        //vector containing the 2 potential values of v for each 4 neighboors
        // First dimension tells if f* or g is used, second to tell which neighboor
        gil::vec3f v[2][4] = {{*g_it - g_it[-1],
                         *g_it - g_it[1],
                         *g_it - g_it[-g_step],
                         *g_it - g_it[g_step]},
                        {*f_it - f_it[-1],
                         *f_it - f_it[1],
                         *f_it - f_it[-f_step],
                         *f_it - f_it[f_step]}};
        for (int k = 0; k < 4; ++k) { // for each neighboor
          //Use average of gradient in both f* and g
          (*dst_it) += 0.5 * (v[0][k] + v[1][k]);
        }
      }
      if (*bound_it == 255) { // if current pixel is part of the boundary
        // 1st part of the right side of the equation, in the form of adding f* to
        // the 4 neighboors of a boundary pixel
        dst_it[-1] += *f_it;
        dst_it[1] += *f_it;
        dst_it[-dst_step] += *f_it;
        dst_it[dst_step] += *f_it;
      }
    }
  }
  return dst;
}

/**
 * Function to execute one iteration of the iterative Jacobi method, applied to
 * the poisson equation. It calculates the left side of the equation and finds
 * a new |src| to use (put in |dst|), only applied at |mask|'s region and using
 * |b| as the boundary corresponding to the right side of poisson's equation.
 */
void jacobi_iteration(gil::mat_cview<gil::vec3f> src,
                      gil::mat_cview<gil::vec3f> b,
                      gil::mat_cview<uint8_t> mask,
                      gil::mat_view<gil::vec3f> dst) {
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
      if (*mask_it >= 128) { // if part of the region targeted by the mask
        // Find f_p's next value. This is basically the equation 7 presented
        // in http://www.cs.virginia.edu/~connelly/class/2014/comp_photo/proj2/poisson.pdf
        // except the 2nd term of the left side was added on both side, and both
        // sides were devided by |Np| (4.0). Therefore, boundary + f's value for each
        // neighboors, devided by 4
        *dst_it = (*b_it + src_it[-1] + src_it[1] + src_it[-src_step] + src_it[src_step]) / 4.0;
      }
    }
  }
}

/**
 * Deprecated function calculating the 2nd term of the left side of poisson
 * blending's equation.
 */
void apply_remainder(gil::mat_cview<gil::vec3f> src,
                     gil::mat_cview<uint8_t> mask,
                     gil::mat_view<gil::vec3f> dst) {
  size_t src_step = src.stride();
  for (int i = 1; i < src.rows()-1; ++i) {
    const gil::vec3f* src_it = src.row_cbegin(i)+1;
    const uint8_t* mask_it = mask.row_cbegin(i)+1;
    gil::vec3f* dst_it = dst.row_begin(i)+1;
    for (size_t j = 1; j < src.cols()-1; ++j, ++src_it, ++dst_it, ++mask_it) {
      if (*mask_it >= 128) {// if pixel targeted by the mask
        // Sum of all 4 neighboors' f_q value
        *dst_it = (src_it[-1] + src_it[1] + src_it[-src_step] + src_it[src_step]);
      }
    }
  }
}

/**
 * Copies |mask|'s white region's corresponding pixels in |src| to |dst|
 * TODO Make it template for vec3f and vec4f
 */
void copy(gil::mat_cview<gil::vec3f> src,
          gil::mat_cview<uint8_t> mask,
          gil::mat_view<gil::vec3f> dst) {
  assert(src.size() == mask.size());
  assert(dst.size() == mask.size());

  for (int i = 0; i < mask.rows(); ++i) {
    const uint8_t* mask_it = mask.row_cbegin(i);
    const gil::vec3f* src_it = src.row_cbegin(i);
    gil::vec3f* dst_it = dst.row_begin(i);
    for (size_t j = 0; j < mask.cols(); ++j, ++mask_it, ++src_it, ++dst_it) {
      if (*mask_it >= 128) { // If white pixel in mask
        *dst_it = *src_it; // copie source to destination
      }
    }
  }
}

/**
 * Copies |mask|'s white region's corresponding pixels in |src| to |dst|
 * TODO Make it template for vec3f and vec4f
 */
void copy(gil::mat_cview<gil::vec3f> src,
          gil::mat_cview<uint8_t> mask,
          gil::mat_view<gil::vec4f> dst) {
  assert(src.size() == mask.size());
  assert(dst.size() == mask.size());

  for (int i = 0; i < mask.rows(); ++i) {
    const uint8_t* mask_it = mask.row_cbegin(i);
    const gil::vec3f* src_it = src.row_cbegin(i);
    gil::vec4f* dst_it = dst.row_begin(i);
    for (size_t j = 0; j < mask.cols(); ++j, ++mask_it, ++src_it, ++dst_it) {
      if (*mask_it >= 128) {
        *dst_it = gil::saturate_cast<uint8_t>(*src_it);
      }
    }
  }
}
