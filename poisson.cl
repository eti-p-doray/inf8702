
__constant sampler_t sampler =
      CLK_NORMALIZED_COORDS_FALSE
    | CLK_ADDRESS_CLAMP_TO_EDGE
    | CLK_FILTER_NEAREST;
  
__kernel void make_boundary(__read_only image2d_t mask,
                            __write_only image2d_t boundary) {
  const int2 pos = {get_global_id(0), get_global_id(1)};

  const uint4 mid = read_imageui(mask, sampler, (int2)(pos.x, pos.y));
  const uint4 left = read_imageui(mask, sampler, (int2)(pos.x-1, pos.y));
  const uint4 right = read_imageui(mask, sampler, (int2)(pos.x+1, pos.y));
  const uint4 down = read_imageui(mask, sampler, (int2)(pos.x, pos.y-1));
  const uint4 up = read_imageui(mask, sampler, (int2)(pos.x, pos.y+1));
  if ((left[0] >= 128 && mid[0] < 128) ||
      (mid[0] < 128 && right[0] >= 128) ||
      (down[0] >= 128 && mid[0] < 128) ||
      (mid[0] < 128 && up[0] >= 128)) {
    write_imageui(boundary, (int2)(pos.x, pos.y), 255);
  } else {
    write_imageui(boundary, (int2)(pos.x, pos.y), uint4(0));
  }
}

__kernel void make_guidance(__read_only image2d_t f,
                            __read_only image2d_t g,
                            __read_only image2d_t mask,
                            __read_only image2d_t boundary,
                            __write_only image2d_t guidance) {
  const int2 pos = {get_global_id(0), get_global_id(1)};

  float4 res = 0.0;

  const uint4 mask_mid = read_imageui(mask, sampler, (int2)(pos.x, pos.y));
  
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

__kernel void apply_mask(__read_only image2d_t mask,
                         __write_only image2d_t image) {
  const int2 pos = {get_global_id(0), get_global_id(1)};

  const uint4 mask_mid = read_imageui(mask, sampler, (int2)(pos.x, pos.y));
  if (mask_mid[0] < 128) {
    write_imagef(image, (int2)(pos.x, pos.y), float4(0.0));
  }
}

__kernel void jacobi_iteration(__read_only image2d_t src,
                            __read_only image2d_t guidance,
                            __read_only image2d_t mask,
                            __write_only image2d_t dst) {
  const int2 pos = {get_global_id(0), get_global_id(1)};

  float4 res = 0.0;

  const uint4 mask_mid = read_imageui(mask, sampler, (int2)(pos.x, pos.y));
  
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
