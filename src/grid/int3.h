#pragma once

#include <cmath>
#include <limits>
#include "ATen/ATen.h"

struct Int3 {
  int32_t x;
  int32_t y;
  int32_t z;

   Int3() : x(0), y(0), z(0) { }
   Int3(int32_t _x, int32_t _y, int32_t _z) :
      x(_x), y(_y), z(_z) { }

   Int3& operator=(const Int3& other) {
    if (this != &other) {
      this->x = other.x;
      this->y = other.y;
      this->z = other.z;
    }
    return *this;
  }

   Int3& operator+=(const Int3& rhs) {
    this->x += rhs.x;
    this->y += rhs.y;
    this->z += rhs.z;
    return *this;
  }

   const Int3 operator+(const Int3& rhs) const {
    Int3 ret = *this;
    ret += rhs;
    return ret;
  }

   Int3& operator-=(const Int3& rhs) {
    this->x -= rhs.x;
    this->y -= rhs.y;
    this->z -= rhs.z;
    return *this;
  }

   const Int3 operator-(const Int3& rhs) const {
    Int3 ret = *this;
    ret -= rhs;
    return ret;
  }

   const Int3 operator+(const int32_t rhs) const {
    Int3 ret = *this;
    ret.x += rhs;
    ret.y += rhs;
    ret.z += rhs;
    return ret;
  }

   const Int3 operator-(const int32_t rhs) const {
    Int3 ret = *this;
    ret.x -= rhs;
    ret.y -= rhs;
    ret.z -= rhs;
    return ret;
  }

   const Int3 operator*(const int32_t rhs) const {
    Int3 ret = *this;
    ret.x *= rhs;
    ret.y *= rhs;
    ret.z *= rhs;
    return ret;
  }
};

