#include "grid_scalar.h"

GridBase::GridBase(T & grid, bool is_3d) :
     is_3d_(is_3d), tensor_(grid), bckd(grid.type().backend()),
     real(grid.type().scalarType()){
  if (grid.ndimension() != 5) {
    AT_ERROR("GridBase: dim must be 5D (even if simulation is 2D).");
  }

  if (!is_3d_ && zsize() != 1) {
    AT_ERROR("GridBase: 2D grid must have zsize == 1.");
  }
}

T GridBase::getSize() const {
   T size = getType(bckd, at::kInt).tensor({3});
   size[0] = xsize();
   size[1] = ysize();
   size[2] = zsize();
   return size;
}                                           


float GridBase::getDx() const {
  const int32_t size_max = std::max(xsize(), std::max(ysize(), zsize()));
  return static_cast<float>(1) / static_cast<float>(size_max);
}

bool GridBase::isInBounds(const T& p, int bnd) const {
  if (!isIntegralType(p.type().scalarType()) ){
   AT_ERROR("GridBase: isInBound input p is not integral type");
  }

  bool ret = toBool((p[0] >= bnd).__and__(p[1] >= bnd).__and__
              (p[0] < xsize() - bnd).__and__(p[1] < ysize() - bnd));
  if (is_3d_) {
    ret &= toBool((p[0] >= bnd).__and__(p[2] < zsize() - bnd));
  } else {
    ret &= toBool(p[2] == 0);
  }
  return ret; 
}

std::ostream& operator<<(std::ostream& os , const GridBase& outGrid) {
  os << (outGrid.tensor_);
}

// Build index is used in interpol and interpolComponent. It allows to 
// perform interpolation of values in a cell.
void GridBase::buildIndex(T& posi, T& stf0, T& stf1,
                          const T& pos) const {
  
 if (!isIntegralType(posi.type().scalarType()) ){
   AT_ERROR("GridBase: buildIndex posi is not Int type");
  }
 if (!isFloatingType(pos.type().scalarType()) ){
   AT_ERROR("GridBase: buildIndex pos is not floating type");
  }
 if (!isFloatingType(stf0.type().scalarType()) ){
   AT_ERROR("GridBase: buildIndex stf0 is not floating type");
  }
 if (!isFloatingType(stf1.type().scalarType()) ){
   AT_ERROR("GridBase: buildIndex stf1 is not floating type");
  }

 // 0.5 is defined as the center of the first cell as the scheme shows:
 //   |----x----|----x----|----x----|
 //  x=0  0.5   1   1.5   2   2.5   3
  T p = getType(bckd, real).tensor({3});
  p = pos - 0.5;
 // we perform a static cast in ATen, equivalent to truncation towards 0
 // (for both pos and neg values)
  posi = p.toType(getType(bckd,at::kInt));
  stf1 = p - posi.toType(getType(bckd,real));
  stf0 = 1 - stf1;
 
  // Clamp to border.
  if (toBool(p[0] < 0) ) {
    posi[0] = (int) 0;
    stf0[0] = 1.;
    stf1[0] = 0.;
  }
  if (toBool(p[1] < 0) ) {
    posi[1] = (int) 0;
    stf0[1] = 1.;
    stf1[1] = 0.;
  }
  if (toBool(p[2] < 0) ) {
    posi[2] = (int) 0;
    stf0[2] = 1.;
    stf1[2] = 0.;
  }
  if (toBool(posi[0] >= xsize() -1)) {
    posi[0] = (int)(xsize() - 2);
    stf0[0] = 0;
    stf1[0] = 1.;
  }
  if (toBool(posi[1] >= xsize() -1)) {
    posi[1] = (int)(xsize() - 2);
    stf0[1] = 0.;
    stf1[1] = 1.;
  }
  if (toBool(posi[2] >= xsize() -1)) {
    posi[2] = (int)(xsize() - 2);
    stf0[2] = 0.;
    stf1[2] = 1.;
  }
}

// ****************************************************************************
// Flag Grid
// *****************************************************************************
FlagGrid::FlagGrid(T & grid, bool is_3d) :
    GridBase(grid, is_3d) {
  if (nchan() != 1) {
    AT_ERROR("FlagGrid: nchan must be 1 (scalar).");
  }
}

// ****************************************************************************
// Real Grid
// *****************************************************************************

RealGrid::RealGrid(T & grid, bool is_3d) :
    GridBase(grid, is_3d) {
  if (nchan() != 1) {
    AT_ERROR("RealGrid: nchan must be 1 (scalar).");
  }
}

T RealGrid::getInterpolatedHi(const T& pos,
                        int32_t order, T b) const {
  if (!isFloatingType(pos.type().scalarType()) ){
   AT_ERROR("RealGrid: getInterpolatedHi input pos is not floating type");
  }

  T at_zero = getType(bckd,at::kInt).scalarTensor(0);

  switch (order) {
  case 1:
    return interpol(pos, b);
  case 2:
    AT_ERROR("getInterpolatedHi ERROR: cubic not supported.");
    break;
  default:
    AT_ERROR("getInterpolatedHi ERROR: order not supported.");
    break;
  }
  return at_zero;
}

T RealGrid::getInterpolatedWithFluidHi(
    const FlagGrid& flags, const T& pos,
    int32_t order, T b) const {
  if (!isFloatingType(pos.type().scalarType()) ){
   AT_ERROR("RealGrid: getInterpolatedWithFluidHi input pos is not floating type");
  }

  T at_zero = getType(bckd,real).scalarTensor(0);

  switch (order) {
  case 1:
    return interpolWithFluid(flags, pos, b);
  case 2:
    AT_ERROR("getInterpolatedWithFluidHi ERROR: cubic not supported.");
    break;
  default:
    AT_ERROR("getInterpolatedWithFluidHi ERROR: order not supported.");
    break;
  }
  return at_zero;
}

T RealGrid::interpol(const T& pos, T b) const {
  T xi = getType(bckd, at::kInt).tensor({3});
  T stf0 = getType(bckd, real).tensor({3}); 
  T stf1 = getType(bckd, real).tensor({3}); 
  buildIndex(xi, stf0, stf1, pos);

  // xi is the integer positions xi, yi, zi
  // stf0 is the interpolant s0, t0, f0
  // stf1 is the interpolant s1, t1, f1
  T at_zero = getType(bckd,at::kInt).scalarTensor(0); 
 
  if (is_3d()) {
    return ((data(xi[0], xi[1], xi[2], at_zero, b) * stf0[1] +
             data(xi[0], xi[1] + 1, xi[2], at_zero, b) * stf1[1]) * stf0[0] 
        + (data(xi[0] + 1, xi[1], xi[2], at_zero, b) * stf0[1] +
           data(xi[0] + 1, xi[1] + 1, xi[2], at_zero, b) * stf1[1]) * stf1[0]) *stf0[2]
        + ((data(xi[0], xi[1], xi[2] + 1, at_zero, b) * stf0[1] +
           data(xi[0], xi[1] + 1, xi[2] + 1, at_zero, b) * stf1[1]) * stf0[0]
        + (data(xi[0] + 1, xi[1], xi[2] + 1, at_zero, b) * stf0[1] +
           data(xi[0] + 1, xi[1] + 1, xi[2] + 1, at_zero, b) * stf1[1]) * stf1[0]) *stf1[2];
  } else {
    return ((data(xi[0], xi[1], at_zero, at_zero, b) * stf0[1] +
             data(xi[0], xi[1] + 1, at_zero, at_zero, b) * stf1[1]) * stf0[0]
       + (data(xi[0] + 1, xi[1], at_zero, at_zero, b) * stf0[1] +
          data(xi[0] + 1, xi[1] + 1, at_zero, at_zero, b) * stf1[1]) * stf1[0]);
  }
}

void RealGrid::interpol1DWithFluid(
    const T val_a, const bool is_fluid_a,
    const T val_b, const bool is_fluid_b,
    const T t_a,   const T t_b,
    bool* is_fluid_ab, T& val_ab) const {
   
  T at_zero = getType(bckd,at::kInt).scalarTensor(0); 
  
  if (!is_fluid_a && !is_fluid_b) {
    val_ab = at_zero;
    *is_fluid_ab = false;
  } else if (!is_fluid_a) {
    val_ab = val_b;
    *is_fluid_ab = true;
  } else if (!is_fluid_b) {
    val_ab = val_a;
    *is_fluid_ab = true;
  } else {
    val_ab = val_a * t_a + val_b * t_b;
    *is_fluid_ab = true;
  }
}

T RealGrid::interpolWithFluid(
    const FlagGrid& flags,const T& pos, 
    T ibatch) const {

  T at_zero = getType(bckd,at::kInt).scalarTensor(0);

  T xi = getType(bckd, at::kInt).tensor({3});
  T stf0 = getType(bckd, real).tensor({3});
  T stf1 = getType(bckd, real).tensor({3});
   
  buildIndex(xi, stf0, stf1, pos);

  if (is_3d()) {
    // val_ab = data(xi, yi, zi, 0, b) * t0 +
    //          data(xi, yi + 1, zi, 0, b) * t1
    const T p_a = getType(bckd, at::kInt).tensor({3});
    const T p_b = getType(bckd, at::kInt).tensor({3});
    p_a[0] = xi;
    p_a[1] = xi;
    p_a[2] = xi;
    
    p_b[0] = xi[0];
    p_b[1] = xi[1] + 1;
    p_b[2] = xi[2];

    bool is_fluid_ab;
    T val_ab;
    interpol1DWithFluid(data(p_a, at_zero, ibatch), flags.isFluid(p_a, ibatch),
                        data(p_b, at_zero, ibatch), flags.isFluid(p_b, ibatch),
                        stf0[1], stf1[1], &is_fluid_ab, val_ab);

    // val_cd = data(xi + 1, yi, zi, at_zero, b) * t0 +
    //          data(xi + 1, yi + 1, zi, at_zero, b) * t1
    
    const T p_c = getType(bckd, at::kInt).tensor({3});
    const T p_d = getType(bckd, at::kInt).tensor({3});
   
    p_c[0] = xi[0] + 1;
    p_c[1] = xi[1];
    p_c[2] = xi[2];

    p_d[0] = xi[0] + 1;
    p_d[1] = xi[1] + 1;
    p_d[2] = xi[2];

    bool is_fluid_cd;
    T val_cd;
    interpol1DWithFluid(data(p_c, at_zero, ibatch), flags.isFluid(p_c, ibatch),
                        data(p_d, at_zero, ibatch), flags.isFluid(p_d, ibatch),
                        stf0[1], stf1[2], &is_fluid_cd, val_cd);

    // val_ef = data(xi, yi, zi + 1, at_zero, b) * t0 +
    //          data(xi, yi + 1, zi + 1, at_zero, b) * t1
    const T p_e = getType(bckd, at::kInt).tensor({3});
    const T p_f = getType(bckd, at::kInt).tensor({3});
   
    p_e[0] = xi[0];
    p_e[1] = xi[1];
    p_e[2] = xi[2] + 1;

    p_f[0] = xi[0];
    p_f[1] = xi[1] + 1;
    p_f[2] = xi[2] + 1;
    
    bool is_fluid_ef;
    T val_ef;
    interpol1DWithFluid(data(p_e, at_zero, ibatch), flags.isFluid(p_e, ibatch),
                        data(p_f, at_zero, ibatch), flags.isFluid(p_f, ibatch),
                        stf0[1], stf1[1], &is_fluid_ef, val_ef);

    // val_gh = data(xi + 1, yi, zi + 1, at_zero, b) * t0 +
    //          data(xi + 1, yi + 1, zi + 1, at_zero, b) * t1
    const T p_g = getType(bckd, at::kInt).tensor({3});
    const T p_h = getType(bckd, at::kInt).tensor({3});
   
    p_g[0] = xi[0] + 1;
    p_g[1] = xi[1];
    p_g[2] = xi[2] + 1;

    p_h[0] = xi[0] + 1;
    p_h[1] = xi[1] + 1;
    p_h[2] = xi[2] + 1;
    bool is_fluid_gh;
    T val_gh;
    interpol1DWithFluid(data(p_g, at_zero, ibatch), flags.isFluid(p_g, ibatch),
                        data(p_h, at_zero, ibatch), flags.isFluid(p_h, ibatch),
                        stf0[1], stf1[1], &is_fluid_gh, val_gh);

    // val_abcd = val_ab * s0 + val_cd * s1
    bool is_fluid_abcd;
    T val_abcd;
    interpol1DWithFluid(val_ab, is_fluid_ab,
                        val_cd, is_fluid_cd,
                        stf0[0], stf1[0], &is_fluid_abcd, val_abcd);

    // val_efgh = val_ef * s0 + val_gh * s1
    bool is_fluid_efgh;
    T val_efgh;
    interpol1DWithFluid(val_ef, is_fluid_ef,
                        val_gh, is_fluid_gh,
                        stf0[0], stf1[0], &is_fluid_efgh, val_efgh);

    // val = val_abcd * f0 + val_efgh * f1
    bool is_fluid;
    T val;
    interpol1DWithFluid(val_abcd, is_fluid_abcd,
                        val_efgh, is_fluid_efgh,
                        stf0[2], stf1[2], &is_fluid, val);
    
    if (!is_fluid) {
      // None of the 8 cells were fluid. Just return the regular interp
      // of all cells.
      return interpol(pos, ibatch);
    } else {
      return val;
    }
  } else {
    // val_ab = data(xi, yi, at_zero, at_zero, b) * t0 +
    //          data(xi, yi + 1, at_zero, at_zero, b) * t1
    const T p_a = getType(bckd, at::kInt).tensor({3});
    const T p_b = getType(bckd, at::kInt).tensor({3});
   
    p_a[0] = xi[0];
    p_a[1] = xi[1];

    p_b[0] = xi[0];
    p_b[1] = xi[1] + 1;
    
    bool is_fluid_ab;
    T  val_ab;
    interpol1DWithFluid(data(p_a, at_zero, ibatch), flags.isFluid(p_a, ibatch),
                        data(p_b, at_zero, ibatch), flags.isFluid(p_b, ibatch),
                        stf0[1], stf1[1], &is_fluid_ab, val_ab);

    // val_cd = data(xi + 1, yi, at_zero, at_zero, b) * t0 +
    //          data(xi + 1, yi + 1, at_zero, at_zero, b) * t1
    const T p_c = getType(bckd, at::kInt).tensor({3});
    const T p_d = getType(bckd, at::kInt).tensor({3});
   
    p_c[0] = xi[0] + 1;
    p_c[1] = xi[1];

    p_d[0] = xi[0] + 1;
    p_d[1] = xi[1] + 1;
   
    bool is_fluid_cd;
    T val_cd;
    interpol1DWithFluid(data(p_c, at_zero, ibatch), flags.isFluid(p_c, ibatch),
                        data(p_d, at_zero, ibatch), flags.isFluid(p_d, ibatch),
                        stf0[1], stf1[1], &is_fluid_cd, val_cd);

    // val = val_ab * s0 + val_cd * s1
    bool is_fluid;
    T val;
    interpol1DWithFluid(val_ab, is_fluid_ab,
                        val_cd, is_fluid_cd,
                        stf0[0], stf1[0], &is_fluid, val);

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
// *****************************************************************************

MACGrid::MACGrid(T & grid, bool is_3d) :
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
const T MACGrid::getCentered(T i, T j, T k, T b) const {  
  
  T chan = getType(bckd, at::kInt).arange(3); 
  const T x = getType(bckd, real).tensor({3});
  
  x[0] = 0.5 * (data(i, j, k, chan[0], b) + data(i + 1, j, k, chan[0], b));
  x[1] = 0.5 * (data(i, j, k, chan[1] , b) + data(i, j + 1, k, chan[1], b));
  x[2] = !is_3d() ? getType(bckd, real).scalarTensor(0) :
      0.5 * (data(i, j, k, chan[2] , b) + data(i, j , k + 1, chan[2], b));

  return x;
}

void MACGrid::setSafe(T i, T j, T k, T b,
                                const T& val) {
 
  if (!isFloatingType(val.type().scalarType()) ){
     AT_ERROR("MACGrid: setSafe input vector (val) is not floating type");
  }
 
  T chan = getType(bckd, at::kInt).arange(3);

  data(i, j, k, chan[0], b) = val[0];
  data(i, j, k, chan[1], b) = val[1];
  if (is_3d()) {
    data(i, j, k, chan[2], b) = val[2];
  } else {
    // This is a pedantic sanity check. We shouldn't be trying to set the
    // z component on a 2D MAC Grid with anything but zero. This is to make
    // sure that the end user fully understands what this function does.
    if (toBool(val[2] != 0)) {
      AT_ERROR("MACGrid: setSafe z-component is non-zero for a 2D grid.");
    }
  }
}

T MACGrid::getAtMACX(
    int32_t i, int32_t j, int32_t k, int32_t b) const {
  T v = getType(bckd, real).tensor({3});
  v[0] = data(i, j, k, 0, b);
  v[1] = 0.25 * (data(i, j, k, 1, b) + data(i - 1, j, k, 1, b) +
                      data(i, j + 1, k, 1, b) + data(i - 1, j + 1, k, 1, b));
  if (is_3d()) {
    v[2] = 0.25* (data(i, j, k, 2, b) + data(i - 1, j, k, 2, b) +
                       data(i, j, k + 1, 2, b) + data(i - 1, j, k + 1, 2, b));
  } else {
    v[2] = 0;
  }
  return v;
}

T MACGrid::getAtMACY(
    int32_t i, int32_t j, int32_t k, int32_t b) const {
  T v = getType(bckd, real).tensor({3});
  v[0] = 0.25 * (data(i, j, k, 0, b) + data(i, j - 1, k, 0, b) +
                      data(i + 1, j, k, 0, b) + data(i + 1, j - 1, k, 0, b));
  v[1] = data(i, j, k, 1, b);
  if (is_3d()) {
    v[2] = 0.25* (data(i, j, k, 2, b) + data(i, j - 1, k, 2, b) +
                       data(i, j, k + 1, 2, b) + data(i, j - 1, k + 1, 2, b));
  } else { 
    v[2] = 0;
  }
  return v;
}

T MACGrid::getAtMACZ(
    int32_t i, int32_t j, int32_t k, int32_t b) const {
  T v = getType(bckd, real).tensor({3});
  v[0] = 0.25 * (data(i, j, k, 0, b) + data(i, j, k - 1, 0, b) +
                      data(i + 1, j, k, 0, b) + data(i + 1, j, k - 1, 0, b));
  v[1] = 0.25 * (data(i, j, k, 1, b) + data(i, j, k - 1, 1, b) +
                      data(i, j + 1, k, 1, b) + data(i, j + 1, k - 1, 1, b));
  if (is_3d()) {
    v[2] = data(i, j, k, 2, b);
  } else {
    v[2] = 0;
  }
  return v;
}

T MACGrid::getInterpolatedComponentHi(
    const T& pos, int32_t order, T c, T b) const {
  T at_zero = getType(bckd, real).scalarTensor(0);
  if (!isFloatingType(pos.type().scalarType()) ){
   AT_ERROR("MACGrid: getInterpolatedComponentHi input pos is not float");
  }

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
  return at_zero;
}

T MACGrid::interpolComponent(
    const T& pos, T c, T b) const {
  T at_zero = getType(bckd, at::kInt).scalarTensor(0);
  T xi = getType(bckd, at::kInt).tensor({3});
  T stf0 = getType(bckd, real).tensor({3});
  T stf1 = getType(bckd, real).tensor({3});
  
  buildIndex(xi, stf0, stf1, pos);

  if (is_3d()) {
    return ((data(xi[0], xi[1], xi[2], c, b) * stf0[1] +
             data(xi[0], xi[1] + 1, xi[2], c, b) * stf1[1]) * stf0[0]
        + (data(xi[0] + 1, xi[1], xi[2], c, b) * stf0[1] +
           data(xi[0] + 1, xi[1] + 1, xi[2], c, b) * stf1[1]) * stf1[0]) * stf0[2]
        + ((data(xi[0], xi[1], xi[2] + 1, c, b) * stf0[1] +
            data(xi[0], xi[1] + 1, xi[2] + 1, c, b) * stf1[1]) * stf0[0]
        + (data(xi[0] + 1, xi[1], xi[2] + 1, c, b) * stf0[1] +
           data(xi[0] + 1, xi[1] + 1, xi[2] + 1, c, b) * stf1[1]) * stf1[0]) * stf1[2];
  } else {
     return ((data(xi[0], xi[1], at_zero, c, b) * stf0[1] +
              data(xi[0], xi[1] + 1, at_zero, c, b) * stf1[1]) * stf0[0]
        + (data(xi[0] + 1, xi[1], at_zero, c, b) * stf0[1] +
           data(xi[0] + 1, xi[1] + 1, at_zero, c, b) * stf1[1]) * stf1[0]);
  }
}

// ****************************************************************************
// Vec Grid
// *****************************************************************************

VecGrid::VecGrid(T & grid, bool is_3d) :
    GridBase(grid, is_3d) {
  if (nchan() != 2 && nchan() != 3) {
    AT_ERROR("VecGrid: input tensor size[0] is not 2 or 3");
  }
  if (!is_3d && zsize() != 1) {
    AT_ERROR("VecGrid: 2D tensor does not have zsize == 1");
  }
}

void VecGrid::setSafe(int32_t i, int32_t j, int32_t k, int32_t b,
                                const T& val) {
  if (!isFloatingType(val.type().scalarType()) ){
   AT_ERROR("VecGrid: setSafe input val is not floating type");
  }
 
  data(i, j, k, 0, b) = val[0];
  data(i, j, k, 1, b) = val[1];
  if (is_3d()) {
    data(i, j, k, 2, b) = val[2];
  } else {
    // This is a pedantic sanity check. We shouldn't be trying to set the
    // z component on a 2D Vec Grid with anything but zero. This is to make
    // sure that the end user fully understands what this function does.
    if ( toBool(val[2] != 0) ) {
      AT_ERROR("VecGrid: setSafe z-component is non-zero for a 2D grid.");
    }
  }
}

void VecGrid::set(int32_t i, int32_t j, int32_t k, int32_t b,
                            const T& val) {
  data(i, j, k, 0, b) = val[0];
  data(i, j, k, 1, b) = val[1];
  if (is_3d()) {
    data(i, j, k, 2, b) = val[2];
  }
}

// Note: you CANNOT call curl on the border of the grid (if you do then
// the data(...) calls will throw an error.
// Also note that curl in 2D is a scalar, but we will return a vector anyway
// with the scalar value being in the 3rd dim.
T VecGrid::curl(int32_t i, int32_t j, int32_t k,
                                       int32_t b) {
   T v = getType(bckd, real).zeros({3});
   v[2] = 0.5 * ((data(i + 1, j, k, 1, b) -
                  data(i - 1, j, k, 1, b)) -
                 (data(i, j + 1, k, 0, b) -
                  data(i, j - 1, k, 0, b)));
   if(is_3d()) {
      v[0] = 0.5 * ((data(i, j + 1, k, 2, b) -
                     data(i, j - 1, k, 2, b)) -
                    (data(i, j, k + 1, 1, b) -
                     data(i, j, k - 1, 1, b)));
      v[1] = 0.5 * ((data(i, j, k + 1, 0, b) -
                     data(i, j, k - 1, 0, b)) -
                    (data(i + 1, j, k, 2, b) -
                     data(i - 1, j, k, 2, b)));
  }
  return v;
}
