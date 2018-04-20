#include <cmath>
#include <limits>

#include "ATen/ATen.h"
#include "int3.h"

struct vec3 {
  constexpr static const float kEpsilon = 1e-6f;

  float x;
  float y;
  float z;

  vec3() : x(0), y(0), z(0) { }
  vec3(float _x, float _y, float _z) : x(_x), y(_y), z(_z) { }

  vec3& operator=(const vec3& other) {  // Copy assignment.
    if (this != &other) {
      this->x = other.x;
      this->y = other.y;
      this->z = other.z;
    }
    return *this;
  }

  vec3& operator+=(const vec3& rhs) {  // accum vec
    this->x += rhs.x;
    this->y += rhs.y;
    this->z += rhs.z;
    return *this;
  }
 
  const vec3 operator+(const vec3& rhs) const {  // add vec
    vec3 ret = *this;
    ret += rhs;
    return ret;
  }

  vec3& operator-=(const vec3& rhs) {  // neg accum vec
    this->x -= rhs.x;
    this->y -= rhs.y;
    this->z -= rhs.z;
    return *this; 
  }
  
  const vec3 operator-(const vec3& rhs) const {  // sub vec
    vec3 ret = *this;
    ret -= rhs; 
    return ret;
  }

  const vec3 operator+(const float rhs) const {  // add scalar
    vec3 ret = *this;
    ret.x += rhs;
    ret.y += rhs;
    ret.z += rhs;
    return ret;
  }

  const vec3 operator-(const float rhs) const {  // sub scalar
    vec3 ret = *this;
    ret.x -= rhs;
    ret.y -= rhs;
    ret.z -= rhs;
    return ret;
  }  
  
  const vec3 operator*(const float rhs) const {  // mult scalar
    vec3 ret = *this;
    ret.x *= rhs;
    ret.y *= rhs;
    ret.z *= rhs;
    return ret;
  }

  const vec3 operator/(const float rhs) const {  // mult scalar
    vec3 ret = *this;
    ret.x /= rhs;
    ret.y /= rhs;
    ret.z /= rhs;
    return ret;
  }

  inline float& operator()(int64_t i) {
    switch (i) {
    case 0:
      return this->x;
    case 1:
      return this->y;
    case 2:
      return this->z;
    default:
      AT_ERROR("vec3 out of bounds.");
      exit(-1);
      break;
    }
  }

  inline float operator()(int64_t i) const {
    return (*this)(i);
  }

  inline float norm() const {
    const float length_sq =
        this->x * this->x + this->y * this->y + this->z * this->z;
    if (length_sq > static_cast<float>(kEpsilon)) {
      return std::sqrt(length_sq);
    } else {
      return static_cast<float>(0);
    }
  }

  inline void normalize() {
    const float norm = this->norm();
    if (norm > static_cast<float>(kEpsilon)) {
      this->x /= norm;
      this->y /= norm;
      this->z /= norm;
    } else {
      this->x = 0;
      this->y = 0;
      this->z = 0;
    }
  }

  static vec3 cross(const vec3& a,
                              const vec3& b) {
    vec3 ret;
    ret.x = (a.y * b.z) - (a.z * b.y);
    ret.y = (a.z * b.x) - (a.x * b.z);
    ret.z = (a.x * b.y) - (a.y * b.x);
    return ret;
  }
};

Int3 toInt3(const vec3& val);
