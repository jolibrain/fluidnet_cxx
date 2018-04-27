#include "vec3.h"

Int3 toInt3(const vec3& val) {
  return Int3(static_cast<int32_t>(val.x),
              static_cast<int32_t>(val.y),
              static_cast<int32_t>(val.z));
};
