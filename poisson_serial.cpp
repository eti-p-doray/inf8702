#include "poisson_serial.hpp"

#include <algorithm>

gil::mat<uint8_t> make_boundary(gil::mat_cview<uint8_t> mask) {
  gil::mat<uint8_t> boundary({mask.rows(), mask.cols()});
  size_t mask_step = mask.stride();
  size_t bound_step = boundary.stride();
  for (int i = 0; i < mask.rows()-1; ++i) {
    auto mask_it = mask.row_cbegin(i);
    auto bound_it = boundary.row_begin(i);
    for (size_t j = 0; j < mask.cols()-1; ++j, ++mask_it, ++bound_it) {
      if (*mask_it < 128 && mask_it[1] >= 128)
        *bound_it = 255;
      if (*mask_it >= 128 && mask_it[1] < 128)
        bound_it[1] = 255;
      if (*mask_it < 128 && mask_it[mask_step] >= 128)
        *bound_it = 255;
      if (*mask_it >= 128 && mask_it[mask_step] < 128)
        bound_it[bound_step] = 255;
    }
  }
  return boundary;
}

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
      if (*mask_it >= 128) {
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

gil::mat<gil::vec3f> make_guidance2(gil::mat_cview<gil::vec3f> f,
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
      if (*mask_it >= 128) {
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
      if (*mask_it >= 128) {
        *dst_it = (*b_it + src_it[-1] + src_it[1] + src_it[-src_step] + src_it[src_step]) / 4.0;
      }
    }
  }
}

void apply_remainder(gil::mat_cview<gil::vec3f> src,
                     gil::mat_cview<uint8_t> mask,
                     gil::mat_view<gil::vec3f> dst) {
  size_t src_step = src.stride();
  for (int i = 1; i < src.rows()-1; ++i) {
    const gil::vec3f* src_it = src.row_cbegin(i)+1;
    const uint8_t* mask_it = mask.row_cbegin(i)+1;
    gil::vec3f* dst_it = dst.row_begin(i)+1;
    for (size_t j = 1; j < src.cols()-1; ++j, ++src_it, ++dst_it, ++mask_it) {
      if (*mask_it >= 128) {
        *dst_it = (src_it[-1] + src_it[1] + src_it[-src_step] + src_it[src_step]);
      }
    }
  }
}

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
      if (*mask_it >= 128) {
        *dst_it = *src_it;
      }
    }
  }
}

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
