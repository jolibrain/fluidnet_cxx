#include "bcs.h"

namespace fluid {

typedef at::Tensor T;

// *****************************************************************************
// setWallBcsForward
// *****************************************************************************

// Enforce boundary conditions on velocity MAC Grid (i.e. set slip components).
// 
// @input U - input vel field (size(2) can be 2 or 3, indicating 2D / 3D)
// @input flags - input occupancy grid

void setWallBcsForward
(
  T& U, T& flags
) {
  // Check arguments.
  AT_ASSERT(U.dim() == 5 && flags.dim() == 5, "Dimension mismatch");
  AT_ASSERT(flags.size(1) == 1, "flags is not scalar");
  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);

  bool is3D = (U.size(1) == 3);
  if (!is3D) {
     AT_ASSERT(d == 1, "2D velocity field but zdepth > 1");
     AT_ASSERT(U.size(1) == 2, "2D velocity field must have only 2 channels");
  }
  AT_ASSERT((U.size(0) == bsz && U.size(2) == d &&
             U.size(3) == h && U.size(4) == w), "Size mismatch");

  AT_ASSERT(U.is_contiguous() && flags.is_contiguous(),
            "Input is not contiguous");

  T i = infer_type(flags).arange(0, w).view({1,w}).expand({bsz, d, h, w}).toType(at::kLong);
  T j = infer_type(i).arange(0, h).view({1,h,1}).expand({bsz, d, h, w});
  T k = zeros_like(i);
  if (is3D) {
     k = infer_type(i).arange(0, d).view({1,d,1,1}).expand({bsz, d, h, w});
  }
  T zero = zeros_like(i);
  T zeroBy = zero.toType(at::kByte);

  T idx_b = infer_type(i).arange(0, bsz).view({bsz,1,1,1});
  idx_b = idx_b.expand({bsz,d,h,w});

  T mCont = ones_like(zeroBy);

  T cur_fluid = flags.eq(TypeFluid).squeeze(1);
  T cur_obs = flags.eq(TypeObstacle).squeeze(1);
  T mNotFluidNotObs = cur_fluid.ne(1).__and__(cur_obs.ne(1));
  mCont.masked_fill_(mNotFluidNotObs, 0);

  T i_l = zero.where( (i <=0), i - 1);
  T obst100 = zeroBy.where( i <= 0, (flags.index({idx_b, zero, k, j, i_l}).eq(TypeObstacle))).__and__(mCont);
  U.select(1,0).masked_fill_(obst100, 0);

  T obs_fluid100 = zeroBy.where( i <= 0, (flags.index({idx_b, zero, k, j, i_l}).eq(TypeFluid))).
   __and__(cur_obs).__and__(mCont);
  U.select(1,0).masked_fill_(obs_fluid100, 0);

  T j_l = zero.where( (j <= 0), j - 1);
  T obst010 = zeroBy.where( j <= 0, (flags.index({idx_b, zero, k, j_l, i}).eq(TypeObstacle))).__and__(mCont);
  U.select(1,1).masked_fill_(obst010, 0);

  T obs_fluid010 = zeroBy.where( j <= 0, (flags.index({idx_b, zero, k, j_l, i}).eq(TypeFluid))).
   __and__(cur_obs).__and__(mCont);
  U.select(1,1).masked_fill_(obs_fluid010, 0);

  if (is3D) {
    T k_l = zero.where( (k <= 0), k - 1);

    T obst001 = zeroBy.where( k <= 0, (flags.index({idx_b, zero, k_l, j, i}).eq(TypeObstacle))).__and__(mCont);
    U.select(1,2).masked_fill_(obst001, 0);

    T obs_fluid001 = zeroBy.where( k <= 0, (flags.index({idx_b, zero, k_l, j, i}).eq(TypeFluid))).
   __and__(cur_obs).__and__(mCont);
    U.select(1,2).masked_fill_(obs_fluid001, 0);
  }

// TODO: implement TypeStick BCs
}

} // namespace fluid
