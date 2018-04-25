#include <iostream>
#include <sstream>
#include <mutex>

#include "grid.h"
#include "stack_trace.h"

GridBase::GridBase(at::Tensor* grid, bool is_3d) :
     is_3d_(is_3d), tensor_(grid), p_grid_(grid->data<float>()) {
  if (grid->ndimension() != 5) {
    AT_ERROR("GridBase: dim must be 5D (even if simulation is 2D).");
  }

  if (!is_3d_ && zsize() != 1) {
    AT_ERROR("GridBase: 2D grid must have zsize == 1.");
  }
}

float GridBase::getDx() const {
  const int32_t size_max = std::max(xsize(), std::max(ysize(), zsize()));
  return static_cast<float>(1) / static_cast<float>(size_max);
}

bool GridBase::isInBounds(const Int3& p, int bnd) const {
  bool ret = (p.x >= bnd && p.y >= bnd && p.x < xsize() - bnd &&
              p.y < ysize() - bnd);
  if (is_3d_) {
    ret &= (p.z >= bnd && p.z < zsize() - bnd);
  } else {
    ret &= (p.z == 0);
  }
  return ret; 
}

bool GridBase::isInBounds(const vec3& p,
                                    int bnd) const {
  return isInBounds(toInt3(p), bnd);
}

int32_t GridBase::index5d(int32_t i, int32_t j, int32_t k,
                                    int32_t c, int32_t b) const {
  if (i >= xsize() || j >= ysize() || k >= zsize() || c >= nchan() ||
      b >= nbatch() || i < 0 || j < 0 || k < 0 || c < 0 || b < 0) {
    std::cout << "Error index5D out of bounds" << std::endl << std::flush;
    std::lock_guard<std::mutex> lock(mutex_);
    std::stringstream ss;
    ss << "GridBase: index4d out of bounds:" << std::endl
       << "  (i, j, k, c, b) = (" << i << ", " << j
       << ", " << k << ", " << c << ", " << b << "), size = (" << xsize()
       << ", " << ysize() << ", " << zsize() << ", " << nchan() 
       << nbatch() << ")";
    std::cerr << ss.str() << std::endl << "Stack trace:" << std::endl;
    PrintStacktrace();
    std::cerr << std::endl;
    AT_ERROR("GridBase: index4d out of bounds");
    return 0;
  }
  return (i * xstride() + j * ystride() + k * zstride() + c * cstride() +
          b * bstride());
}

// Build index is used in interpol and interpolComponent. It allows to 
// perform interpolation of values in a cell.
void GridBase::buildIndex(
    int32_t& xi, int32_t& yi, int32_t& zi, float& s0, float& t0, float& f0,
    float& s1, float& t1, float& f1, const vec3& pos) const {
 // Manta defines 0.5 as the center of the first cell, see this on manta/source/grid.h

  const float px = pos.x - static_cast<float>(0.5);
  const float py = pos.y - static_cast<float>(0.5);
  const float pz = pos.z - static_cast<float>(0.5);
  xi = static_cast<int32_t>(px);
  yi = static_cast<int32_t>(py);
  zi = static_cast<int32_t>(pz);
  s1 = px - static_cast<float>(xi);
  s0 = static_cast<float>(1) - s1;
  t1 = py - static_cast<float>(yi);
  t0 = static_cast<float>(1) - t1;
  f1 = pz - static_cast<float>(zi);
  f0 = static_cast<float>(1) - f1;
  // Clamp to border.
  if (px < static_cast<float>(0)) {
    xi = 0;
    s0 = static_cast<float>(1);
    s1 = static_cast<float>(0);
  }
  if (py < static_cast<float>(0)) {
    yi = 0;
    t0 = static_cast<float>(1);
    t1 = static_cast<float>(0);
  }
  if (pz < static_cast<float>(0)) {
    zi = 0;
    f0 = static_cast<float>(1);
    f1 = static_cast<float>(0);
  }
  if (xi >= xsize() - 1) {
    xi = xsize() - 2;
    s0 = static_cast<float>(0);
    s1 = static_cast<float>(1);
  }
  if (yi >= ysize() - 1) {
    yi = ysize() - 2;
    t0 = static_cast<float>(0);
    t1 = static_cast<float>(1);
  }
  if (zsize() > 1) {
    if (zi >= zsize() - 1) {
      zi = zsize() - 2;
      f0 = static_cast<float>(0);
      f1 = static_cast<float>(1);
    }
  }
}

std::mutex GridBase::mutex_;

// ****************************************************************************
// Flag Grid
//*****************************************************************************
FlagGrid::FlagGrid(at::Tensor* grid, bool is_3d) :
    GridBase(grid, is_3d) {
  if (nchan() != 1) {
    AT_ERROR("FlagGrid: nchan must be 1 (scalar).");
  }
}


// ****************************************************************************
// Float Grid
//*****************************************************************************

FloatGrid::FloatGrid(at::Tensor* grid, bool is_3d) :
    GridBase(grid, is_3d) {
  if (nchan() != 1) {
    AT_ERROR("FloatGrid: nchan must be 1 (scalar).");
  }
}


float FloatGrid::getInterpolatedHi(const vec3& pos,
                                           int32_t order, int32_t b) const {
  switch (order) {
  case 1:
    return interpol(pos, b);
  case 2:
    AT_ERROR("getInterpolatedHi ERROR: cubic not supported.");
    // TODO(tompson): implement this.
    break;
  default:
    AT_ERROR("getInterpolatedHi ERROR: order not supported.");
    break;
  }
  return 0;
}

float FloatGrid::getInterpolatedWithFluidHi(
    const FlagGrid& flags, const vec3& pos,
    int32_t order, int32_t b) const {
  switch (order) {
  case 1:
    return interpolWithFluid(flags, pos, b);
  case 2:
    AT_ERROR("getInterpolatedWithFluidHi ERROR: cubic not supported.");
    // TODO(tompson): implement this.
    break;
  default:
    AT_ERROR("getInterpolatedWithFluidHi ERROR: order not supported.");
    break;
  }
  return 0;
}

float FloatGrid::interpol(const vec3& pos, int32_t b) const {
  int32_t xi, yi, zi;
  float s0, t0, f0, s1, t1, f1;
  buildIndex(xi, yi, zi, s0, t0, f0, s1, t1, f1, pos); 

  if (is_3d()) {
    return ((data(xi, yi, zi, 0, b) * t0 +
             data(xi, yi + 1, zi, 0, b) * t1) * s0 
        + (data(xi + 1, yi, zi, 0, b) * t0 +
           data(xi + 1, yi + 1, zi, 0, b) * t1) * s1) * f0
        + ((data(xi, yi, zi + 1, 0, b) * t0 +
           data(xi, yi + 1, zi + 1, 0, b) * t1) * s0
        + (data(xi + 1, yi, zi + 1, 0, b) * t0 +
           data(xi + 1, yi + 1, zi + 1, 0, b) * t1) * s1) * f1;
  } else {
    return ((data(xi, yi, 0, 0, b) * t0 +
             data(xi, yi + 1, 0, 0, b) * t1) * s0
       + (data(xi + 1, yi, 0, 0, b) * t0 +
          data(xi + 1, yi + 1, 0, 0, b) * t1) * s1);
  }
}

void FloatGrid::interpol1DWithFluid(
    const float val_a, const bool is_fluid_a,
    const float val_b, const bool is_fluid_b,
    const float t_a, const float t_b,
    bool* is_fluid_ab, float* val_ab) {
  if (!is_fluid_a && !is_fluid_b) {
    *val_ab = (float)0;
    *is_fluid_ab = false;
  } else if (!is_fluid_a) {
    *val_ab = val_b;
    *is_fluid_ab = true;
  } else if (!is_fluid_b) {
    *val_ab = val_a;
    *is_fluid_ab = true;
  } else {
    *val_ab = val_a * t_a + val_b * t_b;
    *is_fluid_ab = true;
  }
}

float FloatGrid::interpolWithFluid(
    const FlagGrid& flags, const vec3& pos,
    int32_t ibatch) const {
  int32_t xi, yi, zi;
  float s0, t0, f0, s1, t1, f1;
  buildIndex(xi, yi, zi, s0, t0, f0, s1, t1, f1, pos);

  if (is_3d()) {
    // val_ab = data(xi, yi, zi, 0, b) * t0 +
    //          data(xi, yi + 1, zi, 0, b) * t1
    const Int3 p_a(xi, yi, zi);
    const Int3 p_b(xi, yi + 1, zi);
    bool is_fluid_ab;
    float val_ab;
    interpol1DWithFluid(data(p_a, 0, ibatch), flags.isFluid(p_a, ibatch),
                        data(p_b, 0, ibatch), flags.isFluid(p_b, ibatch),
                        t0, t1, &is_fluid_ab, &val_ab);

    // val_cd = data(xi + 1, yi, zi, 0, b) * t0 +
    //          data(xi + 1, yi + 1, zi, 0, b) * t1
    const Int3 p_c(xi + 1, yi, zi);
    const Int3 p_d(xi + 1, yi + 1, zi);
    bool is_fluid_cd;
    float val_cd;
    interpol1DWithFluid(data(p_c, 0, ibatch), flags.isFluid(p_c, ibatch),
                        data(p_d, 0, ibatch), flags.isFluid(p_d, ibatch),
                        t0, t1, &is_fluid_cd, &val_cd);

    // val_ef = data(xi, yi, zi + 1, 0, b) * t0 +
    //          data(xi, yi + 1, zi + 1, 0, b) * t1
    const Int3 p_e(xi, yi, zi + 1);
    const Int3 p_f(xi, yi + 1, zi + 1);
    bool is_fluid_ef;
    float val_ef;
    interpol1DWithFluid(data(p_e, 0, ibatch), flags.isFluid(p_e, ibatch),
                        data(p_f, 0, ibatch), flags.isFluid(p_f, ibatch),
                        t0, t1, &is_fluid_ef, &val_ef);

    // val_gh = data(xi + 1, yi, zi + 1, 0, b) * t0 +
    //          data(xi + 1, yi + 1, zi + 1, 0, b) * t1
    const Int3 p_g(xi + 1, yi, zi + 1);
    const Int3 p_h(xi + 1, yi + 1, zi + 1);
    bool is_fluid_gh;
    float val_gh;
    interpol1DWithFluid(data(p_g, 0, ibatch), flags.isFluid(p_g, ibatch),
                        data(p_h, 0, ibatch), flags.isFluid(p_h, ibatch),
                        t0, t1, &is_fluid_gh, &val_gh);

    // val_abcd = val_ab * s0 + val_cd * s1
    bool is_fluid_abcd;
    float val_abcd;
    interpol1DWithFluid(val_ab, is_fluid_ab, val_cd, is_fluid_cd,
                        s0, s1, &is_fluid_abcd, &val_abcd);

    // val_efgh = val_ef * s0 + val_gh * s1
    bool is_fluid_efgh;
    float val_efgh;
    interpol1DWithFluid(val_ef, is_fluid_ef, val_gh, is_fluid_gh,
                        s0, s1, &is_fluid_efgh, &val_efgh);

    // val = val_abcd * f0 + val_efgh * f1
    bool is_fluid;
    float val;
    interpol1DWithFluid(val_abcd, is_fluid_abcd, val_efgh, is_fluid_efgh,
                        f0, f1, &is_fluid, &val);
    
    if (!is_fluid) {
      // None of the 8 cells were fluid. Just return the regular interp
      // of all cells.
      return interpol(pos, ibatch);
    } else {
      return val;
    }
  } else {
    // val_ab = data(xi, yi, 0, 0, b) * t0 +
    //          data(xi, yi + 1, 0, 0, b) * t1
    const Int3 p_a(xi, yi, 0);
    const Int3 p_b(xi, yi + 1, 0);
    bool is_fluid_ab;
    float val_ab;
    interpol1DWithFluid(data(p_a, 0, ibatch), flags.isFluid(p_a, ibatch),
                        data(p_b, 0, ibatch), flags.isFluid(p_b, ibatch),
                        t0, t1, &is_fluid_ab, &val_ab);

    // val_cd = data(xi + 1, yi, 0, 0, b) * t0 +
    //          data(xi + 1, yi + 1, 0, 0, b) * t1
    const Int3 p_c(xi + 1, yi, 0);
    const Int3 p_d(xi + 1, yi + 1, 0);
    bool is_fluid_cd;
    float val_cd;
    interpol1DWithFluid(data(p_c, 0, ibatch), flags.isFluid(p_c, ibatch),
                        data(p_d, 0, ibatch), flags.isFluid(p_d, ibatch),
                        t0, t1, &is_fluid_cd, &val_cd);

    // val = val_ab * s0 + val_cd * s1
    bool is_fluid;
    float val;
    interpol1DWithFluid(val_ab, is_fluid_ab, val_cd, is_fluid_cd,
                        s0, s1, &is_fluid, &val);

    if (!is_fluid) {
      // None of the 4 cells were fluid. Just return the regular interp
      // of all cells.
      return interpol(pos, ibatch);
    } else {
      return val;
    }
  }
}

// ****************************************************************************
// MAC Grid
//*****************************************************************************

MACGrid::MACGrid(at::Tensor* grid, bool is_3d) :
    GridBase(grid, is_3d) {
  if (nchan() != 2 && nchan() != 3) {
    AT_ERROR("MACGrid: input tensor size[0] is not 2 or 3");
  }
  if (!is_3d && zsize() != 1) {
    AT_ERROR("MACGrid: 2D tensor does not have zsize == 1");
  }
}

// Note: as per other functions, we DO NOT bounds check getCentered. You must
// not call this method on the edge of the simulation domain.
const vec3 MACGrid::getCentered(
    int32_t i, int32_t j, int32_t k, int32_t b) const {  
  const float x = static_cast<float>(0.5) * (data(i, j, k, 0, b) +
                                           data(i + 1, j, k, 0, b));
  const float y = static_cast<float>(0.5) * (data(i, j, k, 1, b) +
                                           data(i, j + 1, k, 1, b));
  const float z = !is_3d() ? static_cast<float>(0) :
      static_cast<float>(0.5) * (data(i, j, k, 2, b) +
                                data(i, j, k + 1, 2, b));
  return vec3(x, y, z);
}

void MACGrid::setSafe(int32_t i, int32_t j, int32_t k, int32_t b,
                                const vec3& val) {
  data(i, j, k, 0, b) = val.x;
  data(i, j, k, 1, b) = val.y;
  if (is_3d()) {
    data(i, j, k, 2, b) = val.z;
  } else {
    // This is a pedantic sanity check. We shouldn't be trying to set the
    // z component on a 2D MAC Grid with anything but zero. This is to make
    // sure that the end user fully understands what this function does.
    if (val.z != 0) {
      AT_ERROR("MACGrid: setSafe z-component is non-zero for a 2D grid.");
    }
  }
}

vec3 MACGrid::getAtMACX(
    int32_t i, int32_t j, int32_t k, int32_t b) const {
  vec3 v;
  v.x = data(i, j, k, 0, b);
  v.y = (float)0.25 * (data(i, j, k, 1, b) + data(i - 1, j, k, 1, b) +
                      data(i, j + 1, k, 1, b) + data(i - 1, j + 1, k, 1, b));
  if (is_3d()) {
    v.z = (float)0.25* (data(i, j, k, 2, b) + data(i - 1, j, k, 2, b) +
                       data(i, j, k + 1, 2, b) + data(i - 1, j, k + 1, 2, b));
  } else {
    v.z = (float)0;
  }
  return v;
}

vec3 MACGrid::getAtMACY(
    int32_t i, int32_t j, int32_t k, int32_t b) const {
  vec3 v;
  v.x = (float)0.25 * (data(i, j, k, 0, b) + data(i, j - 1, k, 0, b) +
                      data(i + 1, j, k, 0, b) + data(i + 1, j - 1, k, 0, b));
  v.y = data(i, j, k, 1, b);
  if (is_3d()) {
    v.z = (float)0.25* (data(i, j, k, 2, b) + data(i, j - 1, k, 2, b) +
                       data(i, j, k + 1, 2, b) + data(i, j - 1, k + 1, 2, b));
  } else { 
    v.z = (float)0;
  }
  return v;
}

vec3 MACGrid::getAtMACZ(
    int32_t i, int32_t j, int32_t k, int32_t b) const {
  vec3 v;
  v.x = (float)0.25 * (data(i, j, k, 0, b) + data(i, j, k - 1, 0, b) +
                      data(i + 1, j, k, 0, b) + data(i + 1, j, k - 1, 0, b));
  v.y = (float)0.25 * (data(i, j, k, 1, b) + data(i, j, k - 1, 1, b) +
                      data(i, j + 1, k, 1, b) + data(i, j + 1, k - 1, 1, b));
  if (is_3d()) {
    v.z = data(i, j, k, 2, b);
  } else {
    v.z = (float)0;
  }
  return v;
}

float MACGrid::getInterpolatedComponentHi(
    const vec3& pos, int32_t order, int32_t c, int32_t b) const {
  switch (order) {
  case 1:
    return interpolComponent(pos, c, b);
  case 2:
    AT_ERROR("getInterpolatedComponentHi ERROR: cubic not supported.");
    // TODO(tompson): implement this.
    break;
  default:
    AT_ERROR("getInterpolatedComponentHi ERROR: order not supported.");
    break;
  }
  return 0;
}

float MACGrid::interpolComponent(
    const vec3& pos, int32_t c, int32_t b) const {
  int32_t xi, yi, zi;
  float s0, t0, f0, s1, t1, f1;
  buildIndex(xi, yi, zi, s0, t0, f0, s1, t1, f1, pos);

  if (is_3d()) {
    return ((data(xi, yi, zi, c, b) * t0 +
             data(xi, yi + 1, zi, c, b) * t1) * s0
        + (data(xi + 1, yi, zi, c, b) * t0 +
           data(xi + 1, yi + 1, zi, c, b) * t1) * s1) * f0
        + ((data(xi, yi, zi + 1, c, b) * t0 +
            data(xi, yi + 1, zi + 1, c, b) * t1) * s0
        + (data(xi + 1, yi, zi + 1, c, b) * t0 +
           data(xi + 1, yi + 1, zi + 1, c, b) * t1) * s1) * f1;
  } else {
     return ((data(xi, yi, 0, c, b) * t0 +
              data(xi, yi + 1, 0, c, b) * t1) * s0
        + (data(xi + 1, yi, 0, c, b) * t0 +
           data(xi + 1, yi + 1, 0, c, b) * t1) * s1);
  }
}
// ****************************************************************************
// Vec Grid
//*****************************************************************************

VecGrid::VecGrid(at::Tensor* grid, bool is_3d) :
    GridBase(grid, is_3d) {
  if (nchan() != 2 && nchan() != 3) {
    AT_ERROR("VecGrid: input tensor size[0] is not 2 or 3");
  }
  if (!is_3d && zsize() != 1) {
    AT_ERROR("VecGrid: 2D tensor does not have zsize == 1");
  }
}

void VecGrid::setSafe(int32_t i, int32_t j, int32_t k, int32_t b,
                                const vec3& val) {
  data(i, j, k, 0, b) = val.x;
  data(i, j, k, 1, b) = val.y;
  if (is_3d()) {
    data(i, j, k, 2, b) = val.z;
  } else {
    // This is a pedantic sanity check. We shouldn't be trying to set the
    // z component on a 2D Vec Grid with anything but zero. This is to make
    // sure that the end user fully understands what this function does.
    if (val.z != 0) {
      AT_ERROR("VecGrid: setSafe z-component is non-zero for a 2D grid.");
    }
  }
}

void VecGrid::set(int32_t i, int32_t j, int32_t k, int32_t b,
                            const vec3& val) {
  data(i, j, k, 0, b) = val.x;
  data(i, j, k, 1, b) = val.y;
  if (is_3d()) {
    data(i, j, k, 2, b) = val.z;
  }
}

// Note: you CANNOT call curl on the border of the grid (if you do then
// the data(...) calls will throw an error.
// Also note that curl in 2D is a scalar, but we will return a vector anyway
// with the scalar value being in the 3rd dim.
vec3 VecGrid::curl(int32_t i, int32_t j, int32_t k,
                                       int32_t b) {
   vec3 v(0, 0, 0);
   v.z = static_cast<float>(0.5) * ((data(i + 1, j, k, 1, b) -
                                    data(i - 1, j, k, 1, b)) -
                                   (data(i, j + 1, k, 0, b) -
                                    data(i, j - 1, k, 0, b)));
  if(is_3d()) {
      v.x = static_cast<float>(0.5) * ((data(i, j + 1, k, 2, b) -
                                       data(i, j - 1, k, 2, b)) -
                                      (data(i, j, k + 1, 1, b) -
                                       data(i, j, k - 1, 1, b)));
      v.y = static_cast<float>(0.5) * ((data(i, j, k + 1, 0, b) -
                                       data(i, j, k - 1, 0, b)) -
                                      (data(i + 1, j, k, 2, b) -
                                       data(i - 1, j, k, 2, b)));
  }
  return v;
}

