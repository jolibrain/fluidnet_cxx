#include "source_term.h"

namespace fluid {

typedef at::Tensor T;

// *****************************************************************************
// addBuoyancy
// *****************************************************************************

// Add buoyancy force. AddBuoyancy has a dt term.
// Note: Buoyancy is added IN-PLACE.
//
// @input U - vel field (size(2) can be 2 or 3, indicating 2D / 3D)
// @input flags - input occupancy grid
// @input density - scalar density grid.
// @input gravity - 3D vector indicating direction of gravity.
// @input dt - scalar timestep.

void addBuoyancy
(
  T& U, T& flags, T& density, T& gravity,
  const float dt
) {
  // Argument check
  AT_ASSERT(U.dim() == 5 && flags.dim() == 5 && density.dim() == 5,
    "Dimension mismatch");
  AT_ASSERT(flags.size(1) == 1, "flags is not scalar");
  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);

  bool is3D = (U.size(1) == 3);

  int bnd = 1;

  if (!is3D) {
     AT_ASSERT(d == 1, "2D velocity field but zdepth > 1");
     AT_ASSERT(U.size(1) == 2, "2D velocity field must have only 2 channels");
  }
  AT_ASSERT((U.size(0) == bsz && U.size(2) == d &&
             U.size(3) == h && U.size(4) == w), "Size mismatch");
  AT_ASSERT(density.is_same_size(flags), "Size mismatch");

  AT_ASSERT(U.is_contiguous() && flags.is_contiguous() &&
            density.is_contiguous(), "Input is not contiguous");

  AT_ASSERT(gravity.dim() == 1 && gravity.size(0) == 3,
           "Gravity must be a 3D vector (even in 2D)");

  T strength = - gravity * (dt / fluid::ten::getDx(flags));

  T i = infer_type(flags).arange(0, w).view({1,w}).expand({bsz, d, h, w}).toType(at::kLong);
  T j = infer_type(i).arange(0, h).view({1,h,1}).expand({bsz, d, h, w});
  T k = zeros_like(i);
  if (is3D) {
     k = infer_type(i).arange(0, d).view({1,d,1,1}).expand({bsz, d, h, w});
  }
  T zero = zeros_like(i);
  T zeroBy = zero.toType(at::kByte);
  T zero_f = zero.toType(infer_type(density));

  T idx_b = infer_type(i).arange(0, bsz).view({bsz,1,1,1});
  idx_b = idx_b.expand({bsz,d,h,w});

  T maskBorder = (i < bnd).__or__
                 (i > w - 1 - bnd).__or__
                 (j < bnd).__or__
                 (j > h - 1 - bnd);
  if (is3D) {
      maskBorder = maskBorder.__or__(k < bnd).__or__
                                    (k > d - 1 - bnd);
  }
  maskBorder = maskBorder.unsqueeze(1);

  // No buoyancy on the border. Set continue (mCont) to false.
  T mCont = ones_like(zeroBy).unsqueeze(1);
  mCont.masked_fill_(maskBorder, 0);

  T isFluid = flags.eq(TypeFluid).__and__(mCont);
  mCont.masked_fill_(isFluid.ne(1), 0);
  mCont.squeeze_(1);

  T fluid100 = zeroBy.where( i <= 0, (flags.index({idx_b, zero, k, j, i-1}).eq(TypeFluid))).__and__(mCont);
  T factor = 0.5 * strength[0] * (density.squeeze(1) +
             zero_f.where(i <= 0, density.index({idx_b, zero, k, j, i-1})) );
  U.select(1,0).masked_scatter_(fluid100, (U.select(1,0) + factor).masked_select(fluid100));

  T fluid010 = zeroBy.where( j <= 0, (flags.index({idx_b, zero, k, j-1, i}).eq(TypeFluid))).__and__(mCont);
  factor = 0.5 * strength[1] * (density.squeeze(1) +
             zero_f.where( j <= 0, density.index({idx_b, zero, k, j-1, i})) );
  U.select(1,1).masked_scatter_(fluid010, (U.select(1,1) + factor).masked_select(fluid010));

  if (is3D) {
    T fluid001 = zeroBy.where( j <= 0, (flags.index({idx_b, zero, k-1, j, i}).eq(TypeFluid))).__and__(mCont);
    factor = 0.5 * strength[2] * (density.squeeze(1) +
               zero_f.where(k <= 1, density.index({idx_b, zero, k-1, j, i})) );
    U.select(1,2).masked_scatter_(fluid001, (U.select(1,2) + factor).masked_select(fluid001));

  }

}

// *****************************************************************************
// addGravity
// *****************************************************************************

// Add gravity force. It has a dt term.
// Note: gravity is added IN-PLACE.
//
// @input U - vel field (size(2) can be 2 or 3, indicating 2D / 3D)
// @input flags - input occupancy grid
// @input gravity - 3D vector indicating direction of gravity.
// @input dt - scalar timestep.

void addGravity
(
  T& U, T& flags, T& gravity,
  const float dt
) {
  // Argument check
  AT_ASSERT(U.dim() == 5 && flags.dim() == 5, "Dimension mismatch");
  AT_ASSERT(flags.size(1) == 1, "flags is not scalar");
  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);

  bool is3D = (U.size(1) == 3);

  int bnd = 1;
  if (!is3D) {
     AT_ASSERT(d == 1, "2D velocity field but zdepth > 1");
     AT_ASSERT(U.size(1) == 2, "2D velocity field must have only 2 channels");
  }
  AT_ASSERT((U.size(0) == bsz && U.size(2) == d &&
             U.size(3) == h && U.size(4) == w), "Size mismatch");

  AT_ASSERT(U.is_contiguous() && flags.is_contiguous(), "Input is not contiguous");

  AT_ASSERT(gravity.dim() == 1 && gravity.size(0) == 3,
           "Gravity must be a 3D vector (even in 2D)");

  T force = gravity * (dt / fluid::ten::getDx(flags));

  T i = infer_type(flags).arange(0, w).view({1,w}).expand({bsz, d, h, w}).toType(at::kLong);
  T j = infer_type(i).arange(0, h).view({1,h,1}).expand({bsz, d, h, w});
  T k = zeros_like(i);
  if (is3D) {
     k = infer_type(i).arange(0, d).view({1,d,1,1}).expand({bsz, d, h, w});
  }
  T zero = zeros_like(i);
  T zeroBy = zero.toType(at::kByte);
  T zero_f = zero.toType(infer_type(U));

  T idx_b = infer_type(i).arange(0, bsz).view({bsz,1,1,1});
  idx_b = idx_b.expand({bsz,d,h,w});

  T maskBorder = (i < bnd).__or__
                 (i > w - 1 - bnd).__or__
                 (j < bnd).__or__
                 (j > h - 1 - bnd);
  if (is3D) {
      maskBorder = maskBorder.__or__(k < bnd).__or__
                                    (k > d - 1 - bnd);
  }
  maskBorder = maskBorder.unsqueeze(1);

  // No buoyancy on the border. Set continue (mCont) to false.
  T mCont = ones_like(zeroBy).unsqueeze(1);
  mCont.masked_fill_(maskBorder, 0);

  T cur_fluid = flags.eq(TypeFluid).__and__(mCont);
  T cur_empty = flags.eq(TypeEmpty).__and__(mCont);

  T mNotFluidNotEmpt = cur_fluid.ne(1).__and__(cur_empty.ne(1));
  mCont.masked_fill_(mNotFluidNotEmpt, 0);

  mCont.squeeze_(1);

  T fluid100 = (zeroBy.where( i <= 0, (flags.index({idx_b, zero, k, j, i-1}).eq(TypeFluid)))
  .__or__(( zeroBy.where( i <= 0, (flags.index({idx_b, zero, k, j, i-1}).eq(TypeEmpty))))
  .__and__(cur_fluid.squeeze(1)))).__and__(mCont);
  U.select(1,0).masked_scatter_(fluid100, (U.select(1,0) + force[0]).masked_select(fluid100));

  T fluid010 = (zeroBy.where( j <= 0, (flags.index({idx_b, zero, k, j-1, i}).eq(TypeFluid)))
  .__or__(( zeroBy.where( j <= 0, (flags.index({idx_b, zero, k, j-1, i}).eq(TypeEmpty))))
  .__and__(cur_fluid.squeeze(1))) ).__and__(mCont);
  U.select(1,1).masked_scatter_(fluid010, (U.select(1,1) + force[1]).masked_select(fluid010));

  if (is3D) {
    T fluid001 = (zeroBy.where( k <= 0, (flags.index({idx_b, zero, k-1, j, i}).eq(TypeFluid)))
    .__or__(( zeroBy.where( k <= 0, (flags.index({idx_b, zero, k-1, j, i}).eq(TypeEmpty))))
    .__and__(cur_fluid.squeeze(1)))).__and__(mCont);
    U.select(1,2).masked_scatter_(fluid001, (U.select(1,2) + force[2]).masked_select(fluid001));
  }

}

} // namespace fluid
