#include "cl/platform.hpp"

namespace cl {

std::vector<platform> get_platforms() {
  cl_uint size = 0;
  clGetPlatformIDs(0, nullptr, &size);
  std::vector<cl_platform_id> platform_ids(size);
  clGetPlatformIDs(size, platform_ids.data(), nullptr);
  std::vector<platform> platforms(platform_ids.begin(), platform_ids.end());
  return platforms;
}

}
