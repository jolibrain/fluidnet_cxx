#include "advection.h"

#include "advect_type.h"
#include "calc_line_trace.h"
#include "grid/cell_type.h"
#include "grid/grid_new.h"

namespace fluid {

typedef at::Tensor T;

// TODO: eliminate fluid::ten:: namespace

// ****************************************************************************
// Advect Scalar
// ****************************************************************************

// Euler advection with line trace (as in Fluid Net)
T SemiLagrangeEulerFluidNet
(
  T& flags, T& vel, T& src, T& maskBorder,
  float dt, float order_space,
  T& i, T& j, T& k,
  const bool line_trace,
  const bool sample_outside_fluid
) {

  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);

  T ret = zeros_like(src);
  T maskSolid = flags.ne(TypeFluid);
  T maskFluid = flags.eq(TypeFluid);

  AT_ASSERT(maskSolid.equal(1-maskFluid), "Masks are not complementary!");
  // Don't advect solid geometry. 
  ret.masked_scatter_(maskSolid, src.masked_select(maskSolid));
  
  T pos = infer_type(src).zeros({bsz, 3, d, h, w});
 
  pos.select(1,0) = i.toType(infer_type(src)) + 0.5;
  pos.select(1,1) = j.toType(infer_type(src)) + 0.5;
  pos.select(1,2) = k.toType(infer_type(src)) + 0.5;

  T displacement = zeros_like(pos);
 
  // getCentered already eliminates border cells, no need to perform a masked select.
  displacement.masked_scatter_(maskBorder.ne(1), fluid::ten::getCentered(vel));
  displacement.mul_(-dt);
 
  // Calculate a line trace from pos along displacement.
  // NOTE: this is expensive (MUCH more expensive than Manta's routines), but
  // can avoid some artifacts which would occur sampling into Geometry.
 
  T back_pos;
  calcLineTrace(pos, displacement, flags, back_pos, line_trace);
 
  // Finally, sample the field at this back position.
  if (!sample_outside_fluid) {
    ret.masked_scatter_(maskFluid,
         fluid::ten::interpolWithFluid(src, flags, back_pos).masked_select(maskFluid));
  } else {
    ret.masked_scatter_(maskFluid,
         fluid::ten::interpol(src, back_pos).masked_select(maskFluid));
  }
  return ret;
}

// Same kernel as previous one, except that it saves the 
// particle trace position. This is used for the Fluid Net
// MacCormack routine (it does
// a local search around these positions in clamp routine).
T SemiLagrangeEulerFluidNetSavePos 
( 
  T& flags, T& vel, T& src, T& maskBorder, 
  float dt, float order_space, 
  T& i, T& j, T& k, 
  const bool line_trace, 
  const bool sample_outside_fluid, 
  T& pos 
) {

  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);
  bool is3D = (d > 1);

  T ret = zeros_like(src);
  T maskSolid = flags.ne(TypeFluid);
  T maskFluid = flags.eq(TypeFluid);
  AT_ASSERT(maskSolid.equal(1-maskFluid), "Masks are not complementary!");
  
  T start_pos = infer_type(src).zeros({bsz, 3, d, h, w});
 
  start_pos.select(1,0) = i.toType(infer_type(src)) + 0.5;
  start_pos.select(1,1) = j.toType(infer_type(src)) + 0.5;
  start_pos.select(1,2) = k.toType(infer_type(src)) + 0.5;

  // Don't advect solid geometry.
  pos.masked_scatter_(maskSolid, start_pos.masked_select(maskSolid)); 
  ret.masked_scatter_(maskSolid, src.masked_select(maskSolid));

  T displacement = zeros_like(start_pos);
  
  // getCentered already eliminates border cells, no need to perform a masked select.
  displacement.masked_scatter_(maskBorder.ne(1), fluid::ten::getCentered(vel));
  displacement.mul_(-dt);

  // Calculate a line trace from pos along displacement.
  // NOTE: this is expensive (MUCH more expensive than Manta's routines), but
  // can avoid some artifacts which would occur sampling into Geometry.
  T back_pos;
  calcLineTrace(start_pos, displacement, flags, back_pos, line_trace);

  pos.select(1,0).masked_scatter_(maskFluid.squeeze(1) , back_pos.select(1,0).masked_select(maskFluid.squeeze(1))); 
  pos.select(1,1).masked_scatter_(maskFluid.squeeze(1), back_pos.select(1,1).masked_select(maskFluid.squeeze(1))); 
  if (is3D) {
    pos.select(1,2).masked_scatter_(maskFluid.squeeze(1), back_pos.select(1,2).masked_select(maskFluid.squeeze(1))); 
  }
  // Finally, sample the field at this back position.
  if (!sample_outside_fluid) {
    ret.masked_scatter_(maskFluid,
         fluid::ten::interpolWithFluid(src, flags, back_pos).masked_select(maskFluid));
  } else {
    T interp = fluid::ten::interpol(src, back_pos).squeeze(1);
    ret.squeeze(1).masked_scatter_(maskFluid.squeeze(1),
         interp.masked_select(maskFluid.squeeze(1)));
  }
  return ret;
}

T MacCormackCorrect
(
  T& flags, const T& old,
  const T& fwd, const T& bwd,
  const float strength,
  bool is_levelset
)
{
  T dst = fwd.clone();
  T maskFluid = (flags.eq(TypeFluid));
  
  dst.masked_scatter_(maskFluid,
    (dst + strength * 0.5 * (old - bwd)).masked_select(maskFluid));

  return dst;
}

// Clamp routines. It is a search around a the input position
// position for min and max values. If no valid values are found, then
// false (in the mask) is returned (indicating that a clamp shouldn't be performed)
// otherwise true is returned (and the clamp min and max bounds are set). 
T getClampBounds
(
  const T& src, const T& pos, const T& flags,
  const bool sample_outside,
  T& clamp_min, T& clamp_max
)
{
  int bsz = flags.size(0);
  int d   = flags.size(2) ;
  int h   = flags.size(3);
  int w   = flags.size(4);

  T minv = full_like(flags.toType(at::kFloat), INFINITY).squeeze(1);
  T maxv = full_like(flags.toType(at::kFloat), -INFINITY).squeeze(1);
  
  T i0 = infer_type(pos).zeros({bsz, d, h, w}).toType(at::kLong);
  T j0 = infer_type(pos).zeros({bsz, d, h, w}).toType(at::kLong);
  T k0 = infer_type(pos).zeros({bsz, d, h, w}).toType(at::kLong);
 
  i0 = clamp(pos.select(1,0).toType(at::kLong), 0, flags.size(4) - 1);
  j0 = clamp(pos.select(1,1).toType(at::kLong), 0, flags.size(3) - 1);
  k0 = (src.size(1) > 1) ? clamp(pos.select(1,2).toType(at::kLong), 0, flags.size(2) - 1)
        : zeros_like(i0);

  T idx_b = infer_type(i0).arange(0, bsz).view({bsz,1,1,1});
  idx_b = idx_b.expand({bsz,d,h,w});
 
  // We have to look all neighboors.

  T maskOutsideBounds = zeros_like(flags);
  T ncells = zeros_like(flags).squeeze(1);

  T i = zeros_like(i0);
  T j = zeros_like(j0);
  T k = zeros_like(k0);
  T zero = zeros_like(i0);
 
  for (int32_t dk = -1; dk <= 1; dk++) {
    for (int32_t dj = -1; dj <= 1; dj++) {
      for (int32_t di = -1; di<= 1; di++) {
        maskOutsideBounds = (( (k0 + dk) < 0).__or__( (k0 + dk) >= flags.size(2)).__or__
                             ( (j0 + dj) < 0).__or__( (j0 + dj) >= flags.size(3)).__or__
                             ( (i0 + di) < 0).__or__( (i0 + di) >= flags.size(4)));

        i = zero.where( (i0 + di < 0).__or__(i0 + di >= flags.size(4)), i0 + di);
        j = zero.where( (j0 + dj < 0).__or__(j0 + dj >= flags.size(3)), j0 + dj);
        k = zero.where( (k0 + dk < 0).__or__(k0 + dk >= flags.size(2)), k0 + dk);

        T flags_ijk = flags.index({idx_b, zero, k, j, i});
        T src_ijk = src.index({idx_b, zero, k, j, i});
        T maskSample = maskOutsideBounds.ne(1).__and__(flags_ijk.eq(TypeFluid).__or__(sample_outside));

        minv.masked_scatter_(maskSample, at::min(minv, src_ijk).masked_select(maskSample));
        maxv.masked_scatter_(maskSample, at::max(maxv, src_ijk).masked_select(maskSample));
        ncells.masked_scatter_(maskSample, (ncells + 1).masked_select(maskSample));
      }
    }
  }

  T ret = zeros_like(flags).toType(at::kByte);
  ncells = ncells.unsqueeze(1);
  clamp_min.masked_scatter_( (ncells >= 1) , minv.unsqueeze(1).masked_select( ncells >= 1));
  clamp_max.masked_scatter_( (ncells >= 1) , maxv.unsqueeze(1).masked_select( ncells >= 1));
  ret.masked_fill_( (ncells >= 1), 1);

  return ret;
}

T MacCormackClampFluidNet(
  T& flags, T& vel,
  const T& dst, const T& src,
  const T& fwd, const T& fwd_pos,
  const T& bwd_pos, const bool sample_outside 
) {

  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);
  bool is3D = (d > 1);
  // Calculate the clamp bounds.
  T clamp_min = full_like(src, INFINITY);
  T clamp_max = full_like(src, -INFINITY);
  
  // Calculate the clamp bounds around the forward position.
  T pos = infer_type(fwd_pos).zeros({bsz, 3, d, h, w});
  pos.select(1,0) = fwd_pos.select(1,0);
  pos.select(1,1) = fwd_pos.select(1,1);
  if (is3D) {
    pos.select(1,2) = fwd_pos.select(1,2);
  }

  T do_clamp_fwd = getClampBounds(
    src, pos, flags, sample_outside, clamp_min, clamp_max);

  // According to Selle et al. (An Unconditionally Stable MacCormack Method) only
  // a forward search is necessary.
 
  // do_clamp_fwd = false: If the cell is surrounded by fluid neighbors either 
  // in the fwd or backward directions, then we need to revert to an euler step.
  // Otherwise, we found valid values with which to clamp the maccormack corrected
  // quantity. Apply this clamp.
 
  return fwd.where(do_clamp_fwd.ne(1), at::min( clamp_max, at::max(clamp_min, dst)));
}

// Advect scalar field 'p' by the input vel field 'u'.
// 
// @input dt - timestep (seconds).
// @input s - input scalar field to advect
// @input U - input vel field (size(2) can be 2 or 3, indicating 2D / 3D)
// @input flags - input occupancy grid
// @input sDst - Returned scalar field.
// @input method - OPTIONAL - "eulerFluidNet", "maccormackFluidNet"
// @param boundaryWidth - OPTIONAL - boundary width. (default 1)
// @param sampleOutsideFluid - OPTIONAL - For density advection we do not want
// to advect values inside non-fluid cells and so this should be set to false.
// For other quantities (like temperature), this should be true.
// @param maccormackStrength - OPTIONAL - (default 0.75) A strength parameter
// will make the advection eularian (with values interpolating in between). A
// value of 1 (which implements the update from An Unconditionally Stable
// MaCormack Method) tends to add too much high-frequency detail
void advectScalar
(
  float dt, T& src, T& U, T& flags, T& s_dst,
  const std::string method_str,
  int bnd,
  const bool sample_outside_fluid,
  const float maccormack_strength
) {
  // Check sizes
  AT_ASSERT(src.dim() == 5 && U.dim() == 5 && flags.dim() == 5,
    "Dimension mismatch");
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
  AT_ASSERT(U.is_contiguous() && flags.is_contiguous() &&
            src.is_contiguous(), "Input is not contiguous");
  AT_ASSERT(s_dst.dim() == 5, "Size mismatch");
  AT_ASSERT(s_dst.is_contiguous(), "Input is not contiguous");
  AT_ASSERT(s_dst.is_same_size(src) , "Size mismatch");

  T fwd = zeros_like(src);
  T bwd = zeros_like(src);
  T fwd_pos = zeros_like(U);
  T bwd_pos = zeros_like(U);

  AdvectMethod method = StringToAdvectMethod(method_str);
  const bool is_levelset = false;
  const int order_space = 1;
  const bool line_trace = true;

  T pos_corrected = infer_type(src).zeros({bsz, 3, d, h, w});

  T cur_dst = (method == ADVECT_MACCORMACK_FLUIDNET) ? fwd : s_dst;
  
  T idx_x = infer_type(flags).arange(0, w).view({1,w}).expand({bsz, d, h, w}).toType(at::kLong);
  T idx_y = infer_type(idx_x).arange(0, h).view({1,h,1}).expand({bsz, d, h, w});
  T idx_z = zeros_like(idx_x);
  if (is3D) {
     idx_z = infer_type(idx_x).arange(0, d).view({1,d,1,1}).expand({bsz, d, h, w});
  }

  T maskBorder = (idx_x < bnd).__or__
                 (idx_x > w - 1 - bnd).__or__
                 (idx_y < bnd).__or__
                 (idx_y > h - 1 - bnd);
  if (is3D) {
      maskBorder = maskBorder.__or__(idx_z < bnd).__or__
                                    (idx_z > d - 1 - bnd);
  }
  maskBorder.unsqueeze_(1);
 
  // Manta zeros stuff on the border.
  cur_dst.masked_fill_(maskBorder, 0);
  pos_corrected.select(1,0) = idx_x.toType(infer_type(src)) + 0.5;
  pos_corrected.select(1,1) = idx_y.toType(infer_type(src)) + 0.5;
  pos_corrected.select(1,2) = idx_z.toType(infer_type(src)) + 0.5;

  fwd_pos.select(1,0).masked_scatter_(maskBorder.squeeze(1), pos_corrected.select(1,0).masked_select(maskBorder.squeeze(1)));
  fwd_pos.select(1,1).masked_scatter_(maskBorder.squeeze(1), pos_corrected.select(1,1).masked_select(maskBorder.squeeze(1)));
  if (is3D) {
    fwd_pos.select(1,2).masked_scatter_(maskBorder.squeeze(1), pos_corrected.select(1,2).masked_select(maskBorder.squeeze(1)));
  }
 
  // Forward step.
  T val;
  if (method == ADVECT_EULER_FLUIDNET) {
    val = SemiLagrangeEulerFluidNet(flags, U, src, maskBorder, dt, order_space,
            idx_x, idx_y, idx_z, line_trace, sample_outside_fluid);
  } else if (method == ADVECT_MACCORMACK_FLUIDNET) {
    val = SemiLagrangeEulerFluidNetSavePos(flags, U, src, maskBorder, dt, order_space,
            idx_x, idx_y, idx_z, line_trace, sample_outside_fluid, fwd_pos);
  } else {
    AT_ERROR("Advection method not supported!");
  }

  cur_dst.masked_scatter_(maskBorder.eq(0), val.masked_select(maskBorder.eq(0)));

  if (method != ADVECT_MACCORMACK_FLUIDNET) {
    // We're done. The forward Euler step is already in the output array.
    s_dst = cur_dst;
  } else {
    // Otherwise we need to do the backwards step (which is a SemiLagrange
    // step on the forward data - hence we need to finish the above ops
    // before moving on).
 
    // Manta zeros stuff on the border.
    bwd.masked_fill_(maskBorder, 0);
    pos_corrected.select(1,0) = idx_x.toType(infer_type(src))+ 0.5;
    pos_corrected.select(1,1) = idx_y.toType(infer_type(src))+ 0.5;
    pos_corrected.select(1,2) = idx_z.toType(infer_type(src))+ 0.5;

    bwd_pos.masked_scatter_(maskBorder, pos_corrected.masked_select(maskBorder));
    
    // Backwards step
    if (method == ADVECT_MACCORMACK_FLUIDNET) {
      bwd.masked_scatter_(maskBorder.ne(1),
          SemiLagrangeEulerFluidNetSavePos(flags, U, fwd, maskBorder, -dt, order_space,
          idx_x, idx_y, idx_z, line_trace, sample_outside_fluid, bwd_pos)
          .masked_select(maskBorder.ne(1)));       
    }
    // Now compute the correction.
    s_dst = MacCormackCorrect(flags, src, fwd, bwd, maccormack_strength, is_levelset);
  
    // Now perform the clamping.

    if (method == ADVECT_MACCORMACK_FLUIDNET) {
      s_dst.masked_scatter_(maskBorder.ne(1),
          MacCormackClampFluidNet(flags, U, s_dst, src, fwd, fwd_pos, bwd_pos,
          sample_outside_fluid).masked_select(maskBorder.ne(1))); 
    }
  }


}

// ****************************************************************************
// Advect Velocity
// ***************************************************************************

T SemiLagrangeEulerFluidNetMAC
(
  T& flags, T& vel, T& src, T& mask,
  float dt, float order_space,
  const bool line_trace,
  T& i, T& j, T& k
) {

  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);
  bool is3D = (d == 3);

  T ret = zeros_like(src);
  T zero = zeros_like(src);
  T maskSolid = flags.ne(TypeFluid);
  T maskFluid = flags.eq(TypeFluid);

  AT_ASSERT(maskSolid.equal(1-maskFluid), "Masks are not complementary!");
  // Don't advect solid geometry.
  ret.masked_scatter_(maskSolid, src.masked_select(maskSolid));

  // Get correct velocity at MAC position. 
  // No need to shift xpos etc. as lookup field is also shifted. 

  T pos = infer_type(src).zeros({bsz, 3, d, h, w});

  pos.select(1,0) = i.toType(infer_type(src)) + 0.5;
  pos.select(1,1) = j.toType(infer_type(src)) + 0.5;
  pos.select(1,2) = k.toType(infer_type(src)) + 0.5;

  // FluidNet: We floatly want to clamp to the SMALLEST of the steps in each
  // dimension, however this is OK for now (because doing so would expensive)...
  T xpos;
  calcLineTrace(pos, zero.masked_scatter_(mask, fluid::ten::getAtMACX(vel)) * (-dt), flags, xpos, line_trace);

  const T vx = fluid::ten::interpolComponent(src, xpos, 0);

  T ypos;
  calcLineTrace(pos, zero.masked_scatter_(mask, fluid::ten::getAtMACX(vel)) * (-dt), flags, ypos, line_trace);
  const T vy = fluid::ten::interpolComponent(src, ypos, 1);

  T vz = zeros_like(vy);
  if (is3D) {
    T zpos;
    calcLineTrace(pos, zero.masked_scatter_(mask, fluid::ten::getAtMACX(vel)) * (-dt), flags, zpos, line_trace);
    const T vz = fluid::ten::interpolComponent(src, zpos, 1);
  }

  ret.masked_scatter_(maskFluid, (at::cat({vx, vy, vz}, 1)).masked_select(maskFluid));
  return ret;
}

T MacCormackCorrectMAC
(
  T& flags, const T& old,
  const T& fwd, const T& bwd,
  const float strength,
  T& i, T& j, T& k
) {
  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);
  bool is3D = (d == 3);

  T zero = zeros_like(i);
  T zeroBy = zero.toType(at::kByte);
  T idx_b = infer_type(i).arange(0, bsz).view({bsz,1,1,1}).toType(at::kLong);
  idx_b = idx_b.expand({bsz,d,h,w});

  T skip = infer_type(flags).zeros({bsz, 3, d, h, w}).toType(at::kByte);

  T maskSolid = flags.ne(TypeFluid);
  skip.masked_fill_(maskSolid, 1);

  // This allows to never access negative indexes!
  T mask0 = zeroBy.where( i <= 0, (flags.index({idx_b, zero, k, j, i-1}).ne(TypeFluid)));
  skip.select(1,0).masked_fill_(mask0, 1);

  T mask1 = zeroBy.where( j <= 0, (flags.index({idx_b, zero, k, j-1, i}).ne(TypeFluid)));
  skip.select(1,1).masked_fill_(mask1, 1);

  if (is3D) {
    T mask2 = zeroBy.where( k <= 0, (flags.index({idx_b, zero, k-1, j, i}).ne(TypeFluid)));
    skip.select(1,2).masked_fill_(mask2, 1);
  }

  T dst = infer_type(flags).zeros({bsz, 3, d, h, w});

  const int dim = is3D? 3 : 2;
  for (int c = 0; c < dim; ++c) {
    dst.select(1,c) = fwd.select(1,c).where(skip.select(1,c),
            fwd.select(1,c) + strength * 0.5 * (old.select(1,c) - bwd.select(1,c)));
  }
  return dst;
}

T doClampComponentMAC
(
  int chan,
  const T& flags, const T& dst,
  const T& orig,  const T& fwd,
  const T& pos, const T& vel
) {
  int bsz = flags.size(0);
  int d   = flags.size(2) ;
  int h   = flags.size(3);
  int w   = flags.size(4);
  bool is3D = (d > 1);
  T idx_b = infer_type(flags).arange(0, bsz).view({bsz,1,1,1}).toType(at::kLong);
  idx_b = idx_b.expand({bsz,d,h,w});

  T ret = zeros_like(fwd);

  T minv = full_like(flags.toType(at::kFloat), INFINITY);

  T maxv = full_like(flags.toType(at::kFloat), -INFINITY);
  // forward and backward
  std::vector<T> positions;
  positions.insert(positions.end(), (pos - vel).toType(at::kInt));
  positions.insert(positions.end(), (pos + vel).toType(at::kInt));

  T maskRet = ones_like(flags).toType(at::kByte);

  for (int l = 0; l < 2; ++l) {
    T curr_pos = positions[l];

    // clamp forward lookup to grid
    T i0 = curr_pos.select(1,0).clamp(0, flags.size(4) - 2).toType(at::kLong);
    T j0 = curr_pos.select(1,1).clamp(0, flags.size(3) - 2).toType(at::kLong);
    T k0 = curr_pos.select(1,2).clamp(0,
                      is3D ? (flags.size(2) - 2) : 0).toType(at::kLong);
    T i1 = i0 + 1;
    T j1 = j0 + 1;
    T k1 = (is3D) ? (k0 + 1) : k0;

    int bnd = 0;
    T NotInBounds = (i0 < bnd).__or__
                    (i0 > w - 1 - bnd).__or__
                    (j0 < bnd).__or__
                    (j0 > h - 1 - bnd).__or__
                    (i1 < bnd).__or__
                    (i1 > w - 1 - bnd).__or__
                    (j1 < bnd).__or__
                    (j1 > h - 1 - bnd);
    if (is3D) {
        NotInBounds = NotInBounds.__or__(k0 < bnd).__or__
                                        (k0 > d - 1 - bnd).__or__
                                        (k1 < bnd).__or__
                                        (k1 > d - 1 - bnd);
    }
    // We make sure that we don't get out of bounds in call for index.
    // It does not matter the value we fill in, as long as it stays in bounds
    // (0 is straightforward), it will not be selected thanks to the mask InBounds.
    i0.masked_fill_(NotInBounds, 0);
    j0.masked_fill_(NotInBounds, 0);
    k0.masked_fill_(NotInBounds, 0);
    i1.masked_fill_(NotInBounds, 0);
    j1.masked_fill_(NotInBounds, 0);
    k1.masked_fill_(NotInBounds, 0);
    T c = infer_type(i0).scalarTensor(chan);

    NotInBounds = NotInBounds.unsqueeze(1);
    T InBounds = NotInBounds.ne(1);

    ret.masked_scatter_(NotInBounds, fwd.masked_select(NotInBounds));
    maskRet.masked_fill_(NotInBounds, 0);
    
    // find min/max around source position
    T orig000 = orig.index({idx_b, c, k0, j0, i0}).unsqueeze(1);
    minv.masked_scatter_(InBounds, at::min(minv, orig000).masked_select(InBounds));
    maxv.masked_scatter_(InBounds, at::max(maxv, orig000).masked_select(InBounds));
    
    T orig100 = orig.index({idx_b, c, k0, j0, i1}).unsqueeze(1);
    minv.masked_scatter_(InBounds, at::min(minv, orig100).masked_select(InBounds));
    maxv.masked_scatter_(InBounds, at::max(maxv, orig100).masked_select(InBounds));

    T orig010 = orig.index({idx_b, c, k0, j1, i0}).unsqueeze(1);
    minv.masked_scatter_(InBounds, at::min(minv, orig010).masked_select(InBounds));
    maxv.masked_scatter_(InBounds, at::max(maxv, orig010).masked_select(InBounds));

    T orig110 = orig.index({idx_b, c, k0, j1, i1}).unsqueeze(1);
    minv.masked_scatter_(InBounds, at::min(minv, orig110).masked_select(InBounds));
    maxv.masked_scatter_(InBounds, at::max(maxv, orig110).masked_select(InBounds));

    if (is3D) {
      T orig001 = orig.index({idx_b, c, k1, j0, i0}).unsqueeze(1);
      minv.masked_scatter_(InBounds, at::min(minv, orig001).masked_select(InBounds));
      maxv.masked_scatter_(InBounds, at::max(maxv, orig001).masked_select(InBounds));

      T orig101 = orig.index({idx_b, c, k1, j0, i1}).unsqueeze(1);
      minv.masked_scatter_(InBounds, at::min(minv, orig101).masked_select(InBounds));
      maxv.masked_scatter_(InBounds, at::max(maxv, orig101).masked_select(InBounds));

      T orig011 = orig.index({idx_b, c, k1, j1, i0}).unsqueeze(1);
      minv.masked_scatter_(InBounds, at::min(minv, orig011).masked_select(InBounds));
      maxv.masked_scatter_(InBounds, at::max(maxv, orig011).masked_select(InBounds));

      T orig111 = orig.index({idx_b, c, k1, j1, i1}).unsqueeze(1);
      minv.masked_scatter_(InBounds, at::min(minv, orig111).masked_select(InBounds));
      maxv.masked_scatter_(InBounds, at::max(maxv, orig111).masked_select(InBounds));
    }
  }
  // clamp dst
  ret.masked_scatter_(maskRet, at::max(dst, minv.unsqueeze(1)).masked_select(maskRet));
  ret.masked_scatter_(maskRet, at::min(dst, maxv.unsqueeze(1)).masked_select(maskRet));
 
  return ret;
}

T MacCormackClampMAC
(
  const T& flags, const T& vel, const T& dval,
  const T& orig, const T& fwd, const T& mask,
  float dt,
  const T& i, const T& j, const T& k
) {

  int bsz = flags.size(0);
  int d   = flags.size(2);
  int h   = flags.size(3);
  int w   = flags.size(4);
  bool is3D = (flags.size(2) > 1);

  T zero = infer_type(vel).zeros({bsz, 3, d, h, w});
  T pos = at::cat({i.unsqueeze(1), j.unsqueeze(1), k.unsqueeze(1)}, 1).toType(infer_type(vel));
  T dfwd = fwd.clone();

  // getAtMACX-Y-Z already eliminates border cells. In border cells we set 0 as vel
  // but it will be selected out by mask in advectVel.

  dval.select(1,0) = doClampComponentMAC(0, flags, dval.select(1,0).unsqueeze(1),
                                            orig,  dfwd.select(1,0).unsqueeze(1),
                                            pos,
               zero.masked_scatter_(mask, fluid::ten::getAtMACX(vel)) * dt).squeeze(1);
  dval.select(1,1) = doClampComponentMAC(1, flags, dval.select(1,1).unsqueeze(1),
                                            orig,  dfwd.select(1,1).unsqueeze(1),
                                            pos,
               zero.masked_scatter_(mask, fluid::ten::getAtMACY(vel)) * dt).squeeze(1);
  if (is3D) {
     dval.select(1,2) = doClampComponentMAC(2, flags, dval.select(1,2).unsqueeze(1),
                                               orig,  dfwd.select(1,2).unsqueeze(1),
                                               pos,
               zero.masked_scatter_(mask, fluid::ten::getAtMACZ(vel)) * dt).squeeze(1);

  } else {
     dval.select(1,2).fill_(0);
  }
  return dval;
}

// Advect velocity field 'u' by itself and store in uDst.
// 
// @input dt - timestep (seconds).
// @input U - input vel field (size(2) can be 2 or 3, indicating 2D / 3D)
// @input flags - input occupancy grid
// @input UDst - Returned velocity field.
// @input method - OPTIONAL - "eulerFluidNet", "maccormackFluidNet" (default)
// @input boundaryWidth - OPTIONAL - boundary width. (default 1)
// @input maccormackStrength - OPTIONAL - (default 0.75) A strength parameter
// will make the advection more 1st order (with values interpolating in
// between). A value of 1 (which implements the update from "An Unconditionally
// Stable MaCormack Method") tends to add too much high-frequency detail.
void advectVel
(
  float dt, T& U, T& flags, T& U_dst,
  const std::string method_str,
  int bnd,
  const float maccormack_strength

) {
  // We treat 2D advection as 3D (with depth = 1) and
  // no 'w' component for velocity.

  // Check arguments
  AT_ASSERT(U.dim() == 5 && flags.dim() == 5, "Dimension mismatch");
  AT_ASSERT(flags.size(1) == 1, "flags is not scalar");
  int bsz = flags.size(0);
  int d   = flags.size(2);
  int h   = flags.size(3);
  int w   = flags.size(4);

  bool is3D = (U.size(1) == 3);
  if (!is3D) {
     AT_ASSERT(d == 1, "2D velocity field but zdepth > 1");
     AT_ASSERT(U.size(1) == 2, "2D velocity field must have only 2 channels");
  }
  AT_ASSERT((U.size(0) == bsz && U.size(2) == d &&
             U.size(3) == h && U.size(4) == w), "Size mismatch");

  AT_ASSERT(U.is_contiguous() && flags.is_contiguous(),
            "Input is not contiguous");

  AT_ASSERT(U_dst.dim() == 5, "Size mismatch");
  AT_ASSERT(U_dst.is_contiguous(), "Input is not contiguous");
  AT_ASSERT(U_dst.is_same_size(U), "Size mismatch");

  // We always do self-advection, but we could point to another tensor.
  T orig = U.clone();

  // The maccormack method also needs fwd and bwd temporary arrays.
  T fwd = infer_type(flags).zeros({bsz, U.size(1), d, h, w});
  T bwd = infer_type(flags).zeros({bsz, U.size(1), d, h, w});

  AdvectMethod method = StringToAdvectMethod(method_str);

  const int order_space = 1;
  // A full line trace along every ray is expensive but correct (applies to FluidNet
  // methods only).
  const bool line_trace = true;

  T pos_corrected = infer_type(orig).zeros({bsz, 3, d, h, w});

  T cur_U_dst = (method == ADVECT_MACCORMACK_FLUIDNET) ? fwd : U_dst;

  T idx_x = infer_type(flags).arange(0, w).view({1,w}).expand({bsz, d, h, w}).toType(at::kLong);
  T idx_y = infer_type(idx_x).arange(0, h).view({1,h,1}).expand({bsz, d, h, w});
  T idx_z = zeros_like(idx_x);
  if (is3D) {
     idx_z = infer_type(idx_x).arange(0, d).view({1,d,1,1}).expand({bsz, d, h, w});
  }

  T maskBorder = (idx_x < bnd).__or__
                 (idx_x > w - 1 - bnd).__or__
                 (idx_y < bnd).__or__
                 (idx_y > h - 1 - bnd);
  if (is3D) {
      maskBorder = maskBorder.__or__(idx_z < bnd).__or__
                                    (idx_z > d - 1 - bnd);
  }
  maskBorder = maskBorder.unsqueeze(1);

  // Manta zeros stuff on the border.
  cur_U_dst.select(1,0).masked_scatter_(maskBorder.squeeze(1),
       pos_corrected.select(1,0).masked_select(maskBorder.squeeze(1)));
  cur_U_dst.select(1,1).masked_scatter_(maskBorder.squeeze(1),
       pos_corrected.select(1,1).masked_select(maskBorder.squeeze(1)));
  if (is3D) {
    cur_U_dst.select(1,2).masked_scatter_(maskBorder.squeeze(1),
         pos_corrected.select(1,2).masked_select(maskBorder.squeeze(1)));
  }

  // Forward step.
  T val;
  if (method == ADVECT_EULER_FLUIDNET ||
      method == ADVECT_MACCORMACK_FLUIDNET) {
    val = SemiLagrangeEulerFluidNetMAC(flags, U, orig, maskBorder, dt, order_space,
            line_trace, idx_x, idx_y, idx_z);
  } else {
    AT_ERROR("No defined method for MAC advection");
  }

  // Store in the output array.
  cur_U_dst.select(1,0).masked_scatter_(maskBorder.eq(0).squeeze(1),
       val.select(1,0).masked_select(maskBorder.eq(0).squeeze(1)));
  cur_U_dst.select(1,1).masked_scatter_(maskBorder.eq(0).squeeze(1),
       val.select(1,1).masked_select(maskBorder.eq(0).squeeze(1)));
  if (is3D) {
    cur_U_dst.select(1,2).masked_scatter_(maskBorder.eq(0).squeeze(1),
         val.select(1,2).masked_select(maskBorder.eq(0).squeeze(1)));
  }

  if (method != ADVECT_MACCORMACK_FLUIDNET) {
    // We're done. The forward Euler step is already in the output array.
  } else {
    // Otherwise we need to do the backwards step (which is a SemiLagrange
    // step on the forward data - hence we needed to finish the above loops
    // before moving on).
    bwd.select(1,0).masked_scatter_(maskBorder.squeeze(1),
         pos_corrected.select(1,0).masked_select(maskBorder.squeeze(1)));
    bwd.select(1,1).masked_scatter_(maskBorder.squeeze(1),
         pos_corrected.select(1,1).masked_select(maskBorder.squeeze(1)));
    if (is3D) {
      bwd.select(1,2).masked_scatter_(maskBorder.squeeze(1),
           pos_corrected.select(1,2).masked_select(maskBorder.squeeze(1)));
    }

    // Backward step.
    if (method == ADVECT_MACCORMACK_FLUIDNET) {
      bwd.select(1,0).masked_scatter_(maskBorder.ne(1).squeeze(1),
            SemiLagrangeEulerFluidNetMAC(flags, U, fwd, maskBorder, -dt,
                                         order_space, line_trace, idx_x, idx_y, idx_z)
            .select(1,0).masked_select(maskBorder.ne(1).squeeze(1)));
      bwd.select(1,1).masked_scatter_(maskBorder.ne(1).squeeze(1),
            SemiLagrangeEulerFluidNetMAC(flags, U, fwd, maskBorder, -dt,
                                         order_space, line_trace, idx_x, idx_y, idx_z)
            .select(1,1).masked_select(maskBorder.ne(1).squeeze(1)));
      if (is3D) {
        bwd.select(1,2).masked_scatter_(maskBorder.ne(1).squeeze(1),
              SemiLagrangeEulerFluidNetMAC(flags, U, fwd, maskBorder, -dt,
                                           order_space, line_trace, idx_x, idx_y, idx_z)
              .select(1,2).masked_select(maskBorder.ne(1).squeeze(1)));
      }
    }

    // Now compute the correction.

    U_dst.select(1,0).masked_scatter_(maskBorder.ne(1).squeeze(1),
          MacCormackCorrectMAC(flags, orig, fwd, bwd,
                                       maccormack_strength, idx_x, idx_y, idx_z)
          .select(1,0).masked_select(maskBorder.ne(1).squeeze(1)));
    U_dst.select(1,1).masked_scatter_(maskBorder.ne(1).squeeze(1),
          MacCormackCorrectMAC(flags, orig, fwd, bwd,
                                       maccormack_strength, idx_x, idx_y, idx_z)
          .select(1,1).masked_select(maskBorder.ne(1).squeeze(1)));
    if (is3D) {
      U_dst.select(1,2).masked_scatter_(maskBorder.ne(1).squeeze(1),
            MacCormackCorrectMAC(flags, orig, fwd, bwd,
                                         maccormack_strength, idx_x, idx_y, idx_z)
            .select(1,2).masked_select(maskBorder.ne(1).squeeze(1)));
    }

    // Now perform clamping.
    const T dval = infer_type(U).zeros({bsz, 3, d, h, w});
    dval.select(1,0) = U_dst.select(1,0).clone();
    dval.select(1,1) = U_dst.select(1,1).clone();
    if (is3D) {
      dval.select(1,2) = U_dst.select(1,2).clone();
    }

    U_dst.select(1,0).masked_scatter_(maskBorder.ne(1).squeeze(1),
          MacCormackClampMAC(flags, U, dval, orig, fwd, maskBorder, dt,
                                       idx_x, idx_y, idx_z)
          .select(1,0).masked_select(maskBorder.ne(1).squeeze(1)));
    U_dst.select(1,1).masked_scatter_(maskBorder.ne(1).squeeze(1),
          MacCormackClampMAC(flags, U, dval, orig, fwd, maskBorder, dt,
                                       idx_x, idx_y, idx_z)
          .select(1,1).masked_select(maskBorder.ne(1).squeeze(1)));
    if (is3D) {
      U_dst.select(1,2).masked_scatter_(maskBorder.ne(1).squeeze(1),
            MacCormackClampMAC(flags, U, dval, orig, fwd, maskBorder, dt,
                                         idx_x, idx_y, idx_z)
            .select(1,2).masked_select(maskBorder.ne(1).squeeze(1)));
    }

  }

}
} // namespace fluid
