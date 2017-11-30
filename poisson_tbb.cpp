#include "poisson_serial.hpp"

#include <algorithm>
#include <cmath>

#include <tbb/tbb.h>

#include "poisson_tbb.hpp"

using namespace tbb;

/**
 * Class used by tbb to apply the parallel_for calculating the boundary
 */
class ParallelBoundary {
public:
  ParallelBoundary(const gil::mat_cview<uint8_t> mask, gil::mat_view<uint8_t> boundary)
    : mask_(mask), mask_step_(mask_.stride()), boundary_(boundary) {
    //empty, all in initialisation list
  }

  void operator()(const blocked_range<size_t>& range) const {
    for (size_t i = range.begin(); i != range.end(); ++i ){
      auto mask_it = mask_.row_cbegin(i);
      auto bound_it = boundary_.row_begin(i);
      for (size_t j = 0; j < mask_.cols()-1; ++j, ++mask_it, ++bound_it) {
        if ((mask_it[-1] >= 128 && *mask_it < 128) ||
            (*mask_it < 128 && mask_it[1] >= 128) ||
            (mask_it[-mask_step_] >= 128 && *mask_it < 128) ||
            (*mask_it < 128 && mask_it[mask_step_] >= 128)) {
          *bound_it = 255;
        }
      }
    }
  }

private:
  gil::mat_cview<uint8_t> mask_;
  size_t mask_step_;
  gil::mat_view<uint8_t> boundary_;
};
/**
 * Calculates the boundary delta_omega of the region delimited by |mask|
 */
gil::mat<uint8_t> tbb_make_boundary(gil::mat_cview<uint8_t> mask) {
  gil::mat<uint8_t> boundary({mask.rows(), mask.cols()});
  ParallelBoundary para_bound(mask, boundary);
  parallel_for(blocked_range<size_t>(0, mask.rows() - 1), para_bound);
  return boundary;
}

gil::mat<gil::vec3f> tbb_make_guidance(gil::mat_cview<gil::vec3f> f,
                                   gil::mat_cview<gil::vec3f> g,
                                   gil::mat_cview<uint8_t> mask,
                                   GradientMethod method) {
  // calculate the right side of the equation, see above. Constant across solving
  switch (method) {
    default:
    case GradientMethod::BASE:
      return tbb_make_guidance(f, g, mask, tbb_make_boundary(mask));

    case GradientMethod::MAX_MIXING:
      return tbb_make_guidance_mixed_gradient(f, g, mask, tbb_make_boundary(mask));

    case GradientMethod::AVG_MIXING:
      return tbb_make_guidance_mixed_gradient_avg(f, g, mask, tbb_make_boundary(mask));
  }
}

/**
 * Class used by the parallel_for calculating guidance field.
 */
class ParallelGuidance {
private:
  gil::mat_view<gil::vec3f> guidance_;
  gil::mat_cview<uint8_t> mask_;
  gil::mat_cview<uint8_t> boundary_;
  gil::mat_cview<gil::vec3f> g_;
  gil::mat_cview<gil::vec3f> f_;

  size_t f_step_;
  size_t g_step_;
  size_t bound_step_;
public:
  ParallelGuidance(const gil::mat_cview<gil::vec3f> f, const gil::mat_cview<gil::vec3f> g,
    const gil::mat_cview<uint8_t> mask, const gil::mat_cview<uint8_t> boundary,
    gil::mat_view<gil::vec3f> guidance)
    : mask_(mask), boundary_(boundary), g_(g), f_(f), guidance_(guidance),
      f_step_(f_.stride()), g_step_(g_.stride()), bound_step_(boundary_.stride()){
    // empty, all in initialisation list
  }

  void operator() (const blocked_range<size_t>& range) const {
    for (size_t i = range.begin(); i != range.end(); ++i) {
      auto mask_it = mask_.row_cbegin(i)+1;
      auto bound_it = boundary_.row_cbegin(i)+1;
      auto f_it = f_.row_cbegin(i)+1;
      auto g_it = g_.row_cbegin(i)+1;
      auto guidance_it = guidance_.row_begin(i)+1;
      for (size_t j = 1; j < mask_.cols()-1; ++j, ++mask_it, ++bound_it, ++f_it, ++g_it, ++guidance_it) {
        gil::vec3f temp({0.0f, 0.0f, 0.0f}); //for better quicker accesses
        if (*mask_it >= 128) { // if the current pixel is in the mask
          //2nd summation of the right side of the equation, we calculate the sum of all 4 vectors
          temp += 4.0**g_it - (g_it[-1] + g_it[1] + g_it[-g_step_] + g_it[g_step_]);
        }

        // 1st part of the right side of the equation, in the form of adding f* to
        // the 4 neighboors of a boundary pixel
        if (bound_it[-1] == 255) {
          temp += f_it[-1];
        }
        if (bound_it[1] == 255) {
          temp += f_it[1];
        }
        if (bound_it[-bound_step_] == 255) {
          temp += f_it[-f_step_];
        }
        if (bound_it[bound_step_] == 255) {
          temp += f_it[f_step_];
        }

        *guidance_it += temp;
      }
    }
  }
};

/**
 * Calculates the guidance field composed of the |boundary| in destination image |f|
 * and the vector field corresponding to the |mask|'s area in |g|
 */
gil::mat<gil::vec3f> tbb_make_guidance(gil::mat_cview<gil::vec3f> f,
                                   gil::mat_cview<gil::vec3f> g,
                                   gil::mat_cview<uint8_t> mask,
                                   gil::mat_cview<uint8_t> boundary) {
  assert(f.size() == mask.size());
  assert(g.size() == mask.size());
  assert(boundary.size() == mask.size());
  gil::mat<gil::vec3f> dst(f.size());
  ParallelGuidance para_guide(f, g, mask, boundary, dst);
  parallel_for(blocked_range<size_t>(1, mask.rows()-1), para_guide);
  return dst;
}

/**
 * Class used by the parallel_for calculating guidance field with mixed gradients
 */
class ParallelGuidanceMixed {
private:
  gil::mat_view<gil::vec3f> guidance_;
  gil::mat_cview<uint8_t> mask_;
  gil::mat_cview<uint8_t> boundary_;
  gil::mat_cview<gil::vec3f> g_;
  gil::mat_cview<gil::vec3f> f_;

  size_t f_step_;
  size_t g_step_;
  size_t bound_step_;
public:
  ParallelGuidanceMixed(const gil::mat_cview<gil::vec3f> f, const gil::mat_cview<gil::vec3f> g,
    const gil::mat_cview<uint8_t> mask, const gil::mat_cview<uint8_t> boundary,
    gil::mat_view<gil::vec3f> guidance)
    : mask_(mask), boundary_(boundary), g_(g), f_(f), guidance_(guidance),
      f_step_(f_.stride()), g_step_(g_.stride()), bound_step_(boundary_.stride())
      {}

  void operator() (const blocked_range<size_t>& range) const {
    for (size_t i = range.begin(); i != range.end(); ++i) {
      auto mask_it = mask_.row_cbegin(i)+1;
      auto bound_it = boundary_.row_cbegin(i)+1;
      auto f_it = f_.row_cbegin(i)+1;
      auto g_it = g_.row_cbegin(i)+1;
      auto guidance_it = guidance_.row_begin(i)+1;
      for (size_t j = 1; j < mask_.cols()-1; ++j, ++mask_it, ++bound_it, ++f_it, ++g_it, ++guidance_it) {
        gil::vec3f temp({0.0f, 0.0f, 0.0f}); //for better quicker accesses
        if (*mask_it >= 128) { // if the current pixel is in the mask
          //vector containing the 2 potential values of v for each 4 neighboors
          // First dimension tells if f* or g is used, second to tell which neighboor
          gil::vec3f v[2][4] = {{*g_it - g_it[-1],
                           *g_it - g_it[1],
                           *g_it - g_it[-g_step_],
                           *g_it - g_it[g_step_]},
                          {*f_it - f_it[-1],
                           *f_it - f_it[1],
                           *f_it - f_it[-f_step_],
                           *f_it - f_it[f_step_]}};
          for (int k = 0; k < 4; ++k) { // for each neighboor
            //Use the highest gradient between the one from f* and the one from g
            int x = gil::norm2(v[0][k]) > gil::norm2(v[1][k]) ? 0 : 1;
            temp += v[x][k];
          }
        }

        // 1st part of the right side of the equation, in the form of adding f* to
        // the 4 neighboors of a boundary pixel
        if (bound_it[-1] == 255) {
          temp += f_it[-1];
        }
        if (bound_it[1] == 255) {
          temp += f_it[1];
        }
        if (bound_it[-bound_step_] == 255) {
          temp += f_it[-f_step_];
        }
        if (bound_it[bound_step_] == 255) {
          temp += f_it[f_step_];
        }

        *guidance_it += temp;
      }
    }
  }
};
/**
 * Calculates the guidance field composed of the |boundary| in destination image |f|
 * and the vector field corresponding to the |mask|'s area in |g|.
 * This implementation uses mixed_gradients instead of g_p - g_q, which means that
 * we pick the max between the gradient in source and in destination
 */
gil::mat<gil::vec3f> tbb_make_guidance_mixed_gradient(gil::mat_cview<gil::vec3f> f,
                                    gil::mat_cview<gil::vec3f> g,
                                    gil::mat_cview<uint8_t> mask,
                                    gil::mat_cview<uint8_t> boundary) {
  assert(f.size() == mask.size());
  assert(g.size() == mask.size());
  assert(boundary.size() == mask.size());
  gil::mat<gil::vec3f> dst(f.size());
  ParallelGuidanceMixed para_guide(f, g, mask, boundary, dst);
  parallel_for(blocked_range<size_t>(1, mask.rows()-1), para_guide);
  return dst;
}

/**
 * Class used by the parallel_for calculating guidance field with mixed gradients
 * using average of gradient instead of the classic max of gradients
 */
class ParallelGuidanceMixedAvg {
private:
  gil::mat_view<gil::vec3f> guidance_;
  gil::mat_cview<uint8_t> mask_;
  gil::mat_cview<uint8_t> boundary_;
  gil::mat_cview<gil::vec3f> g_;
  gil::mat_cview<gil::vec3f> f_;

  size_t f_step_;
  size_t g_step_;
  size_t bound_step_;
public:
  ParallelGuidanceMixedAvg(const gil::mat_cview<gil::vec3f> f, const gil::mat_cview<gil::vec3f> g,
    const gil::mat_cview<uint8_t> mask, const gil::mat_cview<uint8_t> boundary,
    gil::mat_view<gil::vec3f> guidance)
    : mask_(mask), boundary_(boundary), g_(g), f_(f), guidance_(guidance),
      f_step_(f_.stride()), g_step_(g_.stride()), bound_step_(boundary_.stride())
      {}

  void operator() (const blocked_range<size_t>& range) const {
    for (size_t i = range.begin(); i != range.end(); ++i) {
      auto mask_it = mask_.row_cbegin(i)+1;
      auto bound_it = boundary_.row_cbegin(i)+1;
      auto f_it = f_.row_cbegin(i)+1;
      auto g_it = g_.row_cbegin(i)+1;
      auto guidance_it = guidance_.row_begin(i)+1;
      for (size_t j = 1; j < mask_.cols()-1; ++j, ++mask_it, ++bound_it, ++f_it, ++g_it, ++guidance_it) {
        gil::vec3f temp({0.0f, 0.0f, 0.0f}); //for better quicker accesses
        if (*mask_it >= 128) { // if the current pixel is in the mask
          //vector containing the 2 potential values of v for each 4 neighboors
          // First dimension tells if f* or g is used, second to tell which neighboor
          gil::vec3f v[2][4] = {{*g_it - g_it[-1],
                           *g_it - g_it[1],
                           *g_it - g_it[-g_step_],
                           *g_it - g_it[g_step_]},
                          {*f_it - f_it[-1],
                           *f_it - f_it[1],
                           *f_it - f_it[-f_step_],
                           *f_it - f_it[f_step_]}};
          for (int k = 0; k < 4; ++k) { // for each neighboor
            //Use average of gradient in both f* and g
            temp += 0.5 * (v[0][k] + v[1][k]);
          }
        }

        // 1st part of the right side of the equation, in the form of adding f* to
        // the 4 neighboors of a boundary pixel
        if (bound_it[-1] == 255) {
          temp += f_it[-1];
        }
        if (bound_it[1] == 255) {
          temp += f_it[1];
        }
        if (bound_it[-bound_step_] == 255) {
          temp += f_it[-f_step_];
        }
        if (bound_it[bound_step_] == 255) {
          temp += f_it[f_step_];
        }

        *guidance_it += temp;
      }
    }
  }
};
/**
 * Calculates the guidance field composed of the |boundary| in destination image |f|
 * and the vector field corresponding to the |mask|'s area in |g|.
 * This implementation uses mixed_gradients instead of g_p - g_q, which means that
 * we pick the max between the gradient in source and in destination
 */
gil::mat<gil::vec3f> tbb_make_guidance_mixed_gradient_avg(gil::mat_cview<gil::vec3f> f,
                                    gil::mat_cview<gil::vec3f> g,
                                    gil::mat_cview<uint8_t> mask,
                                    gil::mat_cview<uint8_t> boundary) {
  assert(f.size() == mask.size());
  assert(g.size() == mask.size());
  assert(boundary.size() == mask.size());
  gil::mat<gil::vec3f> dst(f.size());
  ParallelGuidanceMixedAvg para_guide(f, g, mask, boundary, dst);
  parallel_for(blocked_range<size_t>(1, mask.rows()-1), para_guide);
  return dst;
}

/**
 * Class used by tbb to apply the parallel_for calculating the Jacobi iteration
 */
class ParallelJacobi {
public:
  ParallelJacobi(const gil::mat_cview<gil::vec3f> src, const gil::mat_cview<gil::vec3f> b,
    const gil::mat_cview<uint8_t> mask, gil::mat_view<gil::vec3f> dst)
    : src_(src), b_(b), mask_(mask), dst_(dst), src_step_(src_.stride()) {
    //empty, all in initialisation list
  }

  void operator()(const blocked_range<size_t>& range) const {
    for (size_t i = range.begin(); i != range.end(); ++i) {
      const gil::vec3f* src_it = src_.row_cbegin(i)+1;
      const gil::vec3f* b_it = b_.row_cbegin(i)+1;
      const uint8_t* mask_it = mask_.row_cbegin(i)+1;
      gil::vec3f* dst_it = dst_.row_begin(i)+1;
      for (size_t j = 1; j < src_.cols()-1; ++j, ++src_it, ++dst_it, ++b_it, ++mask_it) {
        if (*mask_it >= 128) { // if part of the region targeted by the mask
          // Find f_p's next value. This is basically the equation 7 presented
          // in http://www.cs.virginia.edu/~connelly/class/2014/comp_photo/proj2/poisson.pdf
          // except the 2nd term of the left side was added on both side, and both
          // sides were devided by |Np| (4.0). Therefore, boundary + f's value for each
          // neighboors, devided by 4
          *dst_it = (*b_it + src_it[-1] + src_it[1] + src_it[-src_step_] + src_it[src_step_]) / 4.0;
        }
      }
    }
  }

private:
  gil::mat_view<gil::vec3f> dst_;
  gil::mat_cview<gil::vec3f> src_;
  gil::mat_cview<gil::vec3f> b_;
  gil::mat_cview<uint8_t> mask_;
  size_t src_step_;
};
/**
 * Function to execute one iteration of the iterative Jacobi method, applied to
 * the poisson equation. It calculates the left side of the equation and finds
 * a new |src| to use (put in |dst|), only applied at |mask|'s region and using
 * |b| as the boundary corresponding to the right side of poisson's equation.
 */
void tbb_jacobi_iteration(gil::mat_cview<gil::vec3f> src,
                      gil::mat_cview<gil::vec3f> b,
                      gil::mat_cview<uint8_t> mask,
                      gil::mat_view<gil::vec3f> dst) {
  assert(src.size() == mask.size());
  assert(b.size() == mask.size());
  assert(dst.size() == mask.size());
  ParallelJacobi para_jacobi(src, b, mask, dst);
  parallel_for(blocked_range<size_t>(1, src.rows()-1), para_jacobi);
}
