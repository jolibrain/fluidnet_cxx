#pragma once

#include <iostream>
#include <sstream>

#include "ATen/ATen.h"
#include "cell_type.h"
#include "bool_conversion.h"

class GridBase {
public:
  explicit GridBase(at::Type & T, at::Tensor & grid, bool is_3d);

  int64_t nbatch() const { return tensor_.size(0); }
  int64_t nchan() const  { return tensor_.size(1); }
  int64_t zsize() const  { return tensor_.size(2); }
  int64_t ysize() const  { return tensor_.size(3); }
  int64_t xsize() const  { return tensor_.size(4); }

  int64_t numel() const { return xsize() * ysize() * zsize() * nchan()
			         * nbatch(); }  

  bool is_3d() const { return is_3d_; }
  Int3 getSize() const { return Int3(xsize(), ysize(), zsize()); } 

  float getDx() const;

  bool isInBounds(const Int3& p, int bnd) const;

  bool isInBounds(const Vec3& p, int bnd) const 
  { return isInBounds(toInt3(p), bnd); }

  friend std::ostream& operator<<(std::ostream& os, const GridBase& outGrid);
  
private:
  at::Tensor tensor_;
  at::Type & T_;
  const bool is_3d_; 

protected:
  // Use operator() methods in child classes to get at data.
  float data(int32_t i, int32_t j, int32_t k, int32_t c, int32_t b) const {
    return at::Scalar(tensor_[b][c][k][j][i]).toFloat();
  }

  float data(const Int3& pos, int32_t c, int32_t b) const {
    return at::Scalar(tensor_[b][c][pos.z][pos.y][pos.x]).toFloat();
  }

  void buildIndex(int32_t& xi, int32_t& yi, int32_t& zi,
                  float& s0, float& t0, float& f0,
                  float& s1, float& t1, float& f1,
                  const Vec3& pos) const;
};

class FlagGrid : public GridBase {
public:
  explicit FlagGrid(at::Type & T, at::Tensor & grid, bool is_3d);

  float operator()(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return data(i, j, k, 0, b);
  }

  bool isFluid(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeFluid;
  }

  bool isFluid(const Int3& pos, int32_t b) const {
    return static_cast<int>(data(pos.x, pos.y, pos.z, 0, b)) & TypeFluid;
  }

  bool isObstacle(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeObstacle;
  }

  bool isObstacle(const Int3& pos, int32_t b) const {
    return static_cast<int>(data(pos.x, pos.y, pos.z, 0, b)) & TypeObstacle;
  }
 
 bool isStick(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeStick;
  }
  
  bool isEmpty(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeEmpty;
  }

  bool isOutflow(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return static_cast<int>(data(i, j, k, 0, b)) & TypeOutflow;
  }

  bool isOutOfDomain(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return (i < 0 || i >= xsize() || j < 0 || j >= ysize() || k < 0 ||
            k >= zsize() || b < 0 || b >= nbatch());
  }
};

class FloatGrid : public GridBase {
public:
  explicit FloatGrid(at::Type & T, at::Tensor & grid, bool is_3d);

  float operator()(int32_t i, int32_t j, int32_t k, int32_t b) const {
    return data(i, j, k, 0, b);
  };

  float getInterpolatedHi(const Vec3& pos, int32_t order, int32_t b) const;
  
  float getInterpolatedWithFluidHi(const FlagGrid& flag, const Vec3& pos,
                                  int32_t order, int32_t b) const;

  float interpol(const Vec3& pos, int32_t b) const;
  float interpolWithFluid(const FlagGrid& flag,
                         const Vec3& pos, int32_t b) const;
private:
  // Interpol1DWithFluid will return:
  // 1. is_fluid = false if a and b are not fluid.
  // 2. is_fluid = true and data(a) if b is not fluid.
  // 3. is_fluid = true and data(b) if a is not fluid.
  // 4. The linear interpolation between data(a) and data(b) if both are fluid.
  static void interpol1DWithFluid(const float val_a, const bool is_fluid_a,
                                  const float val_b, const bool is_fluid_b,
                                  const float t_a, const float t_b,
                                  bool* is_fluid_ab, float* val_ab);
};

/*
class MACGrid : public GridBase {
public:
  explicit MACGrid(at::Tensor* grid, bool is_3d);

  // Note: as per other functions, we DO NOT bounds check getCentered. You must
  // not call this method on the edge of the simulation domain.
  const vec3 getCentered(int32_t i, int32_t j, int32_t k,
                                   int32_t b) const;

  const vec3 getCentered(const vec3 vec, int32_t b) {
    return getCentered((int32_t)vec.x, (int32_t)vec.y, (int32_t)vec.z, b);
  }

  const vec3 operator()(int32_t i, int32_t j,
                                  int32_t k, int32_t b) const {
    vec3 ret;
    ret.x = data(i, j, k, 0, b);
    ret.y = data(i, j, k, 1, b);
    ret.z = !is_3d() ? static_cast<float>(0) : data(i, j, k, 2, b);
    return ret;
  }

  float& operator()(int32_t i, int32_t j, int32_t k, int32_t c, int32_t b) {
    return data(i, j, k, c, b);
  }

  float operator()(int32_t i, int32_t j, int32_t k, int32_t c, int32_t b) const {
    return data(i, j, k, c, b);
  }

  // setSafe will ignore the 3rd component of the input vector if the
  // MACGrid is 2D.
  void setSafe(int32_t i, int32_t j, int32_t k, int32_t b,
               const vec3& val);

  vec3 getAtMACX(int32_t i, int32_t j, int32_t k, int32_t b) const;
  vec3 getAtMACY(int32_t i, int32_t j, int32_t k, int32_t b) const;
  vec3 getAtMACZ(int32_t i, int32_t j, int32_t k, int32_t b) const;

  float getInterpolatedComponentHi(const vec3& pos,
                                  int32_t order, int32_t c, int32_t b) const;
private:
  float interpolComponent(const vec3& pos, int32_t c, int32_t b) const;
};

class VecGrid : public GridBase {
public:
  explicit VecGrid(at::Tensor* grid, bool is_3d);

  const vec3 operator()(int32_t i, int32_t j, int32_t k,
                                  int32_t b) const {
    vec3 ret;
    ret.x = data(i, j, k, 0, b);
    ret.y = data(i, j, k, 1, b);
    ret.z = !is_3d() ? static_cast<float>(0) : data(i, j, k, 2, b);
    return ret;
  }

  float& operator()(int32_t i, int32_t j, int32_t k, int32_t c, int32_t b) {
    return data(i, j, k, c, b);
  }

  float operator()(int32_t i, int32_t j, int32_t k, int32_t c, int32_t b) const {
    return data(i, j, k, c, b);
  }

  // setSafe will ignore the 3rd component of the input vector if the
  // VecGrid is 2D, but check that it is non-zero.
  void setSafe(int32_t i, int32_t j, int32_t k, int32_t b,
               const vec3& val);
  // set will ignore the 3rd component of the input vector if the VecGrid is 2D
  // and it will NOT check that the component is non-zero.
  void set(int32_t i, int32_t j, int32_t k, int32_t b,
           const vec3& val);

  // Note: you CANNOT call curl on the border of the grid (if you do then
  // the data(...) calls will throw an error.
  // Also note that curl in 2D is a scalar, but we will return a vector anyway
  // with the scalar value being in the 3rd dim.
  vec3 curl(int32_t i, int32_t j, int32_t k, int32_t b);
};

*/
