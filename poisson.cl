
__constant sampler_t sampler =
      CLK_NORMALIZED_COORDS_FALSE
    | CLK_ADDRESS_CLAMP_TO_EDGE
    | CLK_FILTER_NEAREST;

// Computes contour of binary |mask| describing a shape (pixels >= 128 are
// assumed to be part of shape) and writes result to |boundary|.
// and writes result to |boundary|
__kernel void make_boundary(__read_only image2d_t mask,
                            __write_only image2d_t boundary) {
  const int2 pos = {get_global_id(0), get_global_id(1)};

  const uint4 mid = read_imageui(mask, sampler, (int2)(pos.x, pos.y));
  const uint4 left = read_imageui(mask, sampler, (int2)(pos.x-1, pos.y));
  const uint4 right = read_imageui(mask, sampler, (int2)(pos.x+1, pos.y));
  const uint4 down = read_imageui(mask, sampler, (int2)(pos.x, pos.y-1));
  const uint4 up = read_imageui(mask, sampler, (int2)(pos.x, pos.y+1));
  //if neighboor pixel is part of the mask
  if ((left[0] >= 128 && mid[0] < 128) ||
      (mid[0] < 128 && right[0] >= 128) ||
      (down[0] >= 128 && mid[0] < 128) ||
      (mid[0] < 128 && up[0] >= 128)) {
    write_imageui(boundary, (int2)(pos.x, pos.y), 255); // add pixel to boundary
  } else {
    write_imageui(boundary, (int2)(pos.x, pos.y), uint4(0));
  }
}

// Computes the guidance field composed of the |boundary| in destination
// image |f| and the vector field corresponding to the |mask|'s area in |g|
__kernel void make_guidance(__read_only image2d_t f,
                            __read_only image2d_t g,
                            __read_only image2d_t mask,
                            __read_only image2d_t boundary,
                            __write_only image2d_t guidance) {
  const int2 pos = {get_global_id(0), get_global_id(1)};

  float4 res = 0.0;

  const uint4 mask_mid = read_imageui(mask, sampler, (int2)(pos.x, pos.y));

  // if the current pixel is in the mask, 2nd summation of the right side of the
  //equation, we calculate the sum of all 4 vectors
  if (mask_mid[0] >= 128) {
    const float4 g_mid = read_imagef(g, sampler, (int2)(pos.x, pos.y));
    const float4 g_left = read_imagef(g, sampler, (int2)(pos.x-1, pos.y));
    const float4 g_right = read_imagef(g, sampler, (int2)(pos.x+1, pos.y));
    const float4 g_down = read_imagef(g, sampler, (int2)(pos.x, pos.y-1));
    const float4 g_up = read_imagef(g, sampler, (int2)(pos.x, pos.y+1));

    res += float4(4.0) * g_mid - (g_left + g_right + g_up + g_down);
  }

  const uint4 bound_left = read_imageui(boundary, sampler, (int2)(pos.x-1, pos.y));
  const uint4 bound_right = read_imageui(boundary, sampler, (int2)(pos.x+1, pos.y));
  const uint4 bound_down = read_imageui(boundary, sampler, (int2)(pos.x, pos.y-1));
  const uint4 bound_up = read_imageui(boundary, sampler, (int2)(pos.x, pos.y+1));

  // if a neighboor pixel is on the boundary, 1st part of the right side of the
  // equation.
  if (bound_left[0] == 255)
    res += read_imagef(f, sampler, (int2)(pos.x-1, pos.y));
  if (bound_right[0] == 255)
    res += read_imagef(f, sampler, (int2)(pos.x+1, pos.y));
  if (bound_down[0] == 255)
    res += read_imagef(f, sampler, (int2)(pos.x, pos.y-1));
  if (bound_up[0] == 255)
    res += read_imagef(f, sampler, (int2)(pos.x, pos.y+1));

  write_imagef(guidance, (int2)(pos.x, pos.y), res);
}

/**
 * Calculates the guidance field composed of the |boundary| in destination image |f|
 * and the vector field corresponding to the |mask|'s area in |g|. Returns its answer
 * in its output parameter |guidance|.
 * This implementation uses mixed_gradients instead of g_p - g_q, which means that
 * we pick the max between the gradient in source and in destination
 */
__kernel void make_guidance_mixed_gradient(__read_only image2d_t f,
                            __read_only image2d_t g,
                            __read_only image2d_t mask,
                            __read_only image2d_t boundary,
                            __write_only image2d_t guidance) {
  const int2 pos = {get_global_id(0), get_global_id(1)};

  float4 res = 0.0;

  // Read mask and neighboors in f
  const uint4 mask_mid = read_imageui(mask, sampler, (int2)(pos.x, pos.y));
  const float4 f_left = read_imagef(f, sampler, (int2)(pos.x-1, pos.y));
  const float4 f_right = read_imagef(f, sampler, (int2)(pos.x+1, pos.y));
  const float4 f_down = read_imagef(f, sampler, (int2)(pos.x, pos.y-1));
  const float4 f_up = read_imagef(f, sampler, (int2)(pos.x, pos.y+1));

  if (mask_mid[0] >= 128) { // If pixel is white in mask, mixing gradients
    // Calculate the 4 differences between the pixel and its 4 neighboors, for both f and g
    const float4 g_mid = read_imagef(g, sampler, (int2)(pos.x, pos.y));
    const float4 g_left_diff = g_mid - read_imagef(g, sampler, (int2)(pos.x-1, pos.y));
    const float4 g_right_diff = g_mid - read_imagef(g, sampler, (int2)(pos.x+1, pos.y));
    const float4 g_down_diff = g_mid - read_imagef(g, sampler, (int2)(pos.x, pos.y-1));
    const float4 g_up_diff = g_mid - read_imagef(g, sampler, (int2)(pos.x, pos.y+1));
    const float4 f_mid = read_imagef(f, sampler, (int2)(pos.x, pos.y));
    const float4 f_left_diff = f_mid - f_left;
    const float4 f_right_diff = f_mid - f_right;
    const float4 f_down_diff = f_mid - f_down;
    const float4 f_up_diff = f_mid - f_up;

    // For all 4 neighboors, add its value in the image (f or g) with the highest distance to mid
    // NB: we use dot product with itself, since it gives square distance and if
    //     square distance of a vector a is higher than another vector b's, it
    //     that distance of a is higher than the distance of b.
    if (dot(g_left_diff, g_left_diff) > dot(f_left_diff, f_left_diff))
      res += g_left_diff;
    else
      res += f_left_diff;
    if (dot(g_right_diff, g_right_diff) > dot(f_right_diff, f_right_diff))
      res += g_right_diff;
    else
      res += f_right_diff;
    if (dot(g_down_diff, g_down_diff) > dot(f_down_diff, f_down_diff))
      res += g_down_diff;
    else
      res += f_down_diff;
    if (dot(g_up_diff, g_up_diff) > dot(f_up_diff, f_up_diff))
      res += g_up_diff;
    else
      res += f_up_diff;
  }

  // Read neighboor's boundaries
  const uint4 bound_left = read_imageui(boundary, sampler, (int2)(pos.x-1, pos.y));
  const uint4 bound_right = read_imageui(boundary, sampler, (int2)(pos.x+1, pos.y));
  const uint4 bound_down = read_imageui(boundary, sampler, (int2)(pos.x, pos.y-1));
  const uint4 bound_up = read_imageui(boundary, sampler, (int2)(pos.x, pos.y+1));

  // if current pixel is next to a boundary pixel, add that boundary pixel's value
  if (bound_left[0] == 255)
    res += f_left;
  if (bound_right[0] == 255)
    res += f_right;
  if (bound_down[0] == 255)
    res += f_down;
  if (bound_up[0] == 255)
    res += f_up;

  // write output to output parameter
  write_imagef(guidance, (int2)(pos.x, pos.y), res);
}

__kernel void apply_mask(__read_only image2d_t mask,
                         __write_only image2d_t image) {
  const int2 pos = {get_global_id(0), get_global_id(1)};

  const uint4 mask_mid = read_imageui(mask, sampler, (int2)(pos.x, pos.y));
  if (mask_mid[0] < 128) {
    write_imagef(image, (int2)(pos.x, pos.y), float4(0.0));
  }
}

/**
 * Function to execute one iteration of the iterative Jacobi method, applied to
 * the poisson equation. It calculates the left side of the equation and finds
 * a new |src| to use (put in |dst|), only applied at |mask|'s region and using
 * |b| as the boundary corresponding to the right side of poisson's equation.
 */
__kernel void jacobi_iteration(__read_only image2d_t src,
                            __read_only image2d_t guidance,
                            __read_only image2d_t mask,
                            __write_only image2d_t dst) {
  const int2 pos = {get_global_id(0), get_global_id(1)};

  float4 res = 0.0;

  const uint4 mask_mid = read_imageui(mask, sampler, (int2)(pos.x, pos.y));

  // if part of the region targeted by the mask
  // Find f_p's next value. This is basically the equation 7 presented
  // in http://www.cs.virginia.edu/~connelly/class/2014/comp_photo/proj2/poisson.pdf
  // except the 2nd term of the left side was added on both side, and both
  // sides were devided by |Np| (4.0). Therefore, boundary + f's value for each
  // neighboors, devided by 4
  if (mask_mid[0] >= 128) {
    const float4 b_mid = read_imagef(guidance, sampler, (int2)(pos.x, pos.y));
    const float4 src_left = read_imagef(src, sampler, (int2)(pos.x-1, pos.y));
    const float4 src_right = read_imagef(src, sampler, (int2)(pos.x+1, pos.y));
    const float4 src_down = read_imagef(src, sampler, (int2)(pos.x, pos.y-1));
    const float4 src_up = read_imagef(src, sampler, (int2)(pos.x, pos.y+1));

    res = (b_mid + src_left + src_right + src_down + src_up) / float4(4.0);
  }

  write_imagef(dst, (int2)(pos.x, pos.y), res);
}
