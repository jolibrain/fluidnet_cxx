#include "fluids_init.h"

namespace fluid {

typedef at::Tensor T;

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
  displacement.masked_scatter_(maskBorder.ne(1), getCentered(vel));
  displacement.mul_(-dt);
 
  // Calculate a line trace from pos along displacement.
  // NOTE: this is expensive (MUCH more expensive than Manta's routines), but
  // can avoid some artifacts which would occur sampling into Geometry.
  T back_pos;
  calcLineTrace(pos, displacement, flags, back_pos,line_trace);
  
  // Finally, sample the field at this back position.
  if (!sample_outside_fluid) {
    ret.masked_scatter_(maskFluid,
         interpolWithFluid(src, flags, back_pos).masked_select(maskFluid));
  } else {
    ret.masked_scatter_(maskFluid,
         interpol(src, back_pos).masked_select(maskFluid));
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

  T displacement = zeros_like(start_pos);

  // getCentered already eliminates border cells, no need to perform a masked select.
  displacement.masked_scatter_(maskBorder.ne(1), (-dt) * getCentered(vel));

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
  
  // Don't advect solid geometry.
  pos.select(1,0).masked_scatter_(maskSolid.squeeze(1), start_pos.select(1,0).masked_select(maskSolid.squeeze(1))); 
  pos.select(1,1).masked_scatter_(maskSolid.squeeze(1), start_pos.select(1,1).masked_select(maskSolid.squeeze(1))); 
  if (is3D) {
     pos.select(1,2).masked_scatter_(maskSolid.squeeze(1), start_pos.select(1,2).masked_select(maskSolid.squeeze(1))); 
  }
  
  ret.masked_scatter_(maskSolid, src.masked_select(maskSolid));

  // Finally, sample the field at this back position.
  if (!sample_outside_fluid) {
    ret.masked_scatter_(maskFluid,
         interpolWithFluid(src, flags, back_pos).masked_select(maskFluid));
  } else {
    ret.masked_scatter_(maskFluid,
         interpol(src, back_pos).masked_select(maskFluid));
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
  dst.masked_scatter_(maskFluid, (dst + strength * 0.5f * (old - bwd)).masked_select(maskFluid));

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

  T minv = full_like(flags.toType(infer_type(src)), INFINITY).squeeze(1);
  T maxv = full_like(flags.toType(infer_type(src)), -INFINITY).squeeze(1);
  
  T i0 = infer_type(pos).zeros({bsz, d, h, w}).toType(at::kLong);
  T j0 = infer_type(pos).zeros({bsz, d, h, w}).toType(at::kLong);
  T k0 = infer_type(pos).zeros({bsz, d, h, w}).toType(at::kLong);
 
  i0 = clamp(pos.select(1,0).toType(at::kLong), 0, flags.size(4) - 1);
  j0 = clamp(pos.select(1,1).toType(at::kLong), 0, flags.size(3) - 1);
  k0 = (src.size(1) > 1) ? 
      clamp(pos.select(1,2).toType(at::kLong), 0, flags.size(2) - 1) : zeros_like(i0);

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
 
  return fwd.where(do_clamp_fwd.ne(1), at::max( clamp_min, at::min(clamp_max, dst)));
}

at::Tensor advectScalar
(
  float dt, T src, T U, T flags,
  const std::string method_str,
  int bnd,
  const bool sample_outside_fluid,
  const float maccormack_strength
) {
  // Size checking done in python side
  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);

  bool is3D = (U.size(1) == 3);

  T s_dst = zeros_like(src);

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
    // before moving on).) 
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

  return s_dst;
}

// ****************************************************************************
// Advect Velocity
// ***************************************************************************

T SemiLagrangeEulerFluidNetMAC
(
  T& flags, T& vel, T& src, T& maskBorder,
  float dt, float order_space,
  const bool line_trace,
  T& i, T& j, T& k
) {

  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);
  bool is3D = (d > 1);

  T zero = zeros_like(src);
  T ret = infer_type(src).zeros({bsz,3,d,h,w});
  T vec3_0 = infer_type(src).zeros({bsz,3,d,h,w});
  T maskSolid = flags.ne(TypeFluid);
  T maskFluid = flags.eq(TypeFluid);

  AT_ASSERT(maskSolid.equal(1-maskFluid), "Masks are not complementary!");

  // Don't advect solid geometry.
  ret.select(1,0).unsqueeze(1).masked_scatter_(
          maskSolid, src.select(1,0).unsqueeze(1).masked_select(maskSolid));
  ret.select(1,0).unsqueeze(1).masked_scatter_(
          maskSolid, src.select(1,1).unsqueeze(1).masked_select(maskSolid));
  if (is3D) {
    ret.select(1,2).unsqueeze(1).masked_scatter_(
            maskSolid, src.select(1,2).unsqueeze(1).masked_select(maskSolid));
  }
  // Get correct velocity at MAC position. 
  // No need to shift xpos etc. as lookup field is also shifted. 
  T pos = infer_type(src).zeros({bsz, 3, d, h, w});

  pos.select(1,0) = i.toType(infer_type(src)) + 0.5;
  pos.select(1,1) = j.toType(infer_type(src)) + 0.5;
  pos.select(1,2) = k.toType(infer_type(src)) + 0.5;

  // FluidNet: We floatly want to clamp to the SMALLEST of the steps in each
  // dimension, however this is OK for now (because doing so would expensive)...
  T xpos;
  calcLineTrace(pos, vec3_0.masked_scatter_(maskBorder.eq(0),
              getAtMACX(vel)) * (-dt), flags, xpos, line_trace);
  const T vx = interpolComponent(src, xpos, 0);

  T ypos;
  calcLineTrace(pos, vec3_0.masked_scatter_(maskBorder.eq(0),
              getAtMACY(vel)) * (-dt), flags, ypos,line_trace);
  const T vy = interpolComponent(src, ypos, 1);

  T vz = zeros_like(vy);
  if (is3D) {
    T zpos;
    calcLineTrace(pos, vec3_0.masked_scatter_(maskBorder.eq(0),
                getAtMACZ(vel)) * (-dt), flags, zpos,line_trace);
    const T vz = interpolComponent(src, zpos, 2);
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
  bool is3D = (d > 1);

  T zero = zeros_like(i);
  T zeroBy = zero.toType(at::kByte);
  T idx_b = infer_type(i).arange(0, bsz).view({bsz,1,1,1}).toType(at::kLong);
  idx_b = idx_b.expand({bsz,d,h,w});

  T skip = infer_type(flags).zeros({bsz, 3, d, h, w}).toType(at::kByte);

  T maskSolid = flags.ne(TypeFluid);
  skip.masked_fill_(maskSolid, 1);

  // This allows to never access negative indexes!
  T mask0 = zeroBy.where(i<=0, (flags.index({idx_b, zero, k, j, i-1}).ne(TypeFluid)));
  skip.select(1,0).masked_fill_(mask0, 1);

  T mask1 = zeroBy.where(j<=0, (flags.index({idx_b, zero, k, j-1, i}).ne(TypeFluid)));
  skip.select(1,1).masked_fill_(mask1, 1);

  if (is3D) {
    T mask2 = zeroBy.where(k<=0, (flags.index({idx_b, zero, k-1, j, i}).ne(TypeFluid)));
    skip.select(1,2).masked_fill_(mask2, 1);
  }

  T dst = infer_type(flags).zeros({bsz, (is3D? 3:2), d, h, w});
  const int dim = is3D? 3 : 2;

  for (int c = 0; c < dim; ++c) {
    dst.select(1,c) = at::where(skip.select(1,c), fwd.select(1,c),
            fwd.select(1,c) + strength * 0.5f * (old.select(1,c) - bwd.select(1,c)));
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

  T minv = full_like(flags.toType(infer_type(dst)), INFINITY);
  T maxv = full_like(flags.toType(infer_type(dst)), -INFINITY);

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
  ret.masked_scatter_(maskRet,
          at::max(at::min(dst, maxv), minv).masked_select(maskRet));
  return ret;
}

T MacCormackClampMAC
(
  const T& flags, const T& vel, const T& dval,
  const T& orig, const T& fwd, const T& maskBorder,
  float dt,
  const T& i, const T& j, const T& k
) {

  int bsz = flags.size(0);
  int d   = flags.size(2);
  int h   = flags.size(3);
  int w   = flags.size(4);
  bool is3D = (d > 1);

  T zero = infer_type(vel).zeros({bsz, 3, d, h, w});
  T pos = at::cat({i.unsqueeze(1), j.unsqueeze(1), k.unsqueeze(1)}, 1).toType(infer_type(vel));
  T dfwd = fwd.clone();

  // getAtMACX-Y-Z already eliminates border cells. In border cells we set 0 as vel
  // but it will be selected out by mask in advectVel.
  dval.select(1,0) = doClampComponentMAC(0, flags, dval.select(1,0).unsqueeze(1),
    orig,  dfwd.select(1,0).unsqueeze(1), pos,
    zero.masked_scatter_(maskBorder.eq(0), getAtMACX(vel)) * dt).squeeze(1);
  
  dval.select(1,1) = doClampComponentMAC(1, flags, dval.select(1,1).unsqueeze(1),
    orig,  dfwd.select(1,1).unsqueeze(1), pos,
   zero.masked_scatter_(maskBorder.eq(0), getAtMACY(vel)) * dt).squeeze(1);
  if (is3D) {
     dval.select(1,2) = doClampComponentMAC(2, flags, dval.select(1,2).unsqueeze(1),
        orig,  dfwd.select(1,2).unsqueeze(1), pos,
        zero.masked_scatter_(maskBorder.eq(0), getAtMACZ(vel)) * dt).squeeze(1);

  } else {
     dval.select(1,2).fill_(0);
  }
  return dval;
}

at::Tensor advectVel
(
  float dt, T U, T flags,
  const std::string method_str,
  int bnd,
  const float maccormack_strength
) {
  // We treat 2D advection as 3D (with depth = 1) and
  // no 'w' component for velocity.

  int bsz = flags.size(0);
  int d   = flags.size(2);
  int h   = flags.size(3);
  int w   = flags.size(4);

  bool is3D = (U.size(1) == 3);

  T U_dst = zeros_like(U);

  // We always do self-advection, but we could point to another tensor.
  T orig = U.clone();

  // The maccormack method also needs fwd and bwd temporary arrays.
  T fwd = infer_type(flags).zeros({bsz, U.size(1), d, h, w});
  T bwd = infer_type(flags).zeros({bsz, U.size(1), d, h, w});

  AdvectMethod method = StringToAdvectMethod(method_str);

  const int order_space = 1;
  // A full line trace along every ray is expensive but correct (applies to FluidNet
  // methods only).
  const bool line_trace = false;

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
    T CorrectMAC = MacCormackCorrectMAC(flags, orig, fwd, bwd, 
                                       maccormack_strength, idx_x, idx_y, idx_z);
    U_dst.select(1,0).masked_scatter_(maskBorder.ne(1).squeeze(1),
          CorrectMAC.select(1,0).masked_select(maskBorder.ne(1).squeeze(1)));
    U_dst.select(1,1).masked_scatter_(maskBorder.ne(1).squeeze(1),
          CorrectMAC.select(1,1).masked_select(maskBorder.ne(1).squeeze(1)));
    if (is3D) {
      U_dst.select(1,2).masked_scatter_(maskBorder.ne(1).squeeze(1),
            CorrectMAC.select(1,2).masked_select(maskBorder.ne(1).squeeze(1)));
    }

    // Now perform clamping.
    const T dval = infer_type(U).zeros({bsz, 3, d, h, w});
    dval.select(1,0) = U_dst.select(1,0).clone();
    dval.select(1,1) = U_dst.select(1,1).clone();
    if (is3D) {
      dval.select(1,2) = U_dst.select(1,2).clone();
    }

    T ClampMAC = MacCormackClampMAC(flags, U, dval, orig, fwd, maskBorder, 
                                    dt, idx_x, idx_y, idx_z);
    U_dst.select(1,0).masked_scatter_(maskBorder.ne(1).squeeze(1),
             ClampMAC.select(1,0).masked_select(maskBorder.ne(1).squeeze(1)));
    U_dst.select(1,1).masked_scatter_(maskBorder.ne(1).squeeze(1),
             ClampMAC.select(1,1).masked_select(maskBorder.ne(1).squeeze(1)));
    if (is3D) {
      U_dst.select(1,2).masked_scatter_(maskBorder.ne(1).squeeze(1),
               ClampMAC.select(1,2).masked_select(maskBorder.ne(1).squeeze(1)));
    }
  }
  return U_dst;
}

std::vector<T> solveLinearSystemJacobi
(
   T flags,
   T div,
   const bool is3D,
   const float p_tol = 1e-5,
   const int max_iter = 1000,
   const bool verbose = false
) {
  // Check arguments.
  T p = zeros_like(flags);
  AT_ASSERT(p.dim() == 5 && flags.dim() == 5 && div.dim() == 5,
             "Dimension mismatch");
  AT_ASSERT(flags.size(1) == 1, "flags is not scalar");
  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);
  int numel = d * h * w;
  AT_ASSERT(p.is_same_size(flags), "size mismatch");
  AT_ASSERT(div.is_same_size(flags), "size mismatch");
  if (!is3D) {
    AT_ASSERT(d == 1, "d > 1 for a 2D domain");
  }

  AT_ASSERT(p.is_contiguous() && flags.is_contiguous() &&
            div.is_contiguous(), "Input is not contiguous");

  T p_prev = infer_type(p).zeros({bsz, 1, d, h, w});
  T p_delta = infer_type(p).zeros({bsz, 1, d, h, w});
  T p_delta_norm = infer_type(p).zeros({bsz});

  if (max_iter < 1) {
     AT_ERROR("At least 1 iteration is needed (maxIter < 1)");
  }

  // Initialize the pressure to zero.
  p.zero_();

  // Start with the output of the next iteration going to pressure.
  T* cur_p = &p;
  T* cur_p_prev = &p_prev;
  //RealGrid* cur_pressure_prev = &pressure_prev;

  T residual;
 // T at_zero = infer_type(tensor_p).scalarTensor(0);

  int64_t iter = 0;
  while (true) {
    const int32_t bnd =1;
    // Kernel: Jacobi Iteration
    T mCont = infer_type(flags).ones({bsz, 1, d, h, w}).toType(at::kByte); // Continue mask

    T idx_x = infer_type(flags).arange(0, w).view({1,w}).expand({bsz, d, h, w}).toType(at::kLong);
    T idx_y = infer_type(idx_x).arange(0, h).view({1,h,1}).expand({bsz, d, h, w});
    T idx_z = zeros_like(idx_x);
    if (is3D) {
       idx_z = infer_type(idx_x).arange(0, d).view({1,d,1,1}).expand({bsz, d, h, w});
    }

    T idx_b = infer_type(flags).arange(0, bsz).view({bsz,1,1,1}).toType(at::kLong);
    idx_b = idx_b.expand({bsz,d,h,w});

    T maskBorder = (idx_x < bnd).__or__
                   (idx_x > w - 1 - bnd).__or__
                   (idx_y < bnd).__or__
                   (idx_y > h - 1 - bnd);
    if (is3D) {
        maskBorder = maskBorder.__or__(idx_z < bnd).__or__
                                      (idx_z > d - 1 - bnd);
    }
    maskBorder.unsqueeze_(1);

    // Zero pressure on the border.
    cur_p->masked_fill_(maskBorder, 0);
    mCont.masked_fill_(maskBorder, 0);

    T maskObstacle = flags.eq(TypeObstacle).__and__(mCont);
    cur_p->masked_fill_(maskObstacle, 0);
    mCont.masked_fill_(maskObstacle, 0);



    T zero_f = at::zeros_like(p); // Floating zero
    T zero_l = at::zeros_like(p).toType(at::kLong); // Long zero (for index)
    T zeroBy = at::zeros_like(p).toType(at::kByte); // Long zero (for index)
    // Otherwise, we are in a fluid or empty cell.
    // First, we get all the neighbors.

    T pC = *cur_p_prev;

    T i_l = zero_l.where( (idx_x <=0), idx_x - 1);
    T p1 = zero_f.
        where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, idx_z, idx_y, i_l})
        .unsqueeze(1));

    T i_r = zero_l.where( (idx_x > w - 1 - bnd), idx_x + 1);
    T p2 = zero_f.
        where(idx_x >= (w - 3 - bnd), (*cur_p_prev).index({idx_b, zero_l, idx_z, idx_y, i_r})
        .unsqueeze(1));

    T j_l = zero_l.where( (idx_y <= 0), idx_y - 1);
    T p3 = zero_f.
        where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, idx_z, j_l, idx_x})
        .unsqueeze(1));
    T j_r = zero_l.where( (idx_y > h - 1 - bnd), idx_y + 1);
    T p4 = zero_f.
        where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, idx_z, j_r, idx_x})
        .unsqueeze(1));

    T k_l = zero_l.where( (idx_z <= 0), idx_z - 1);
    T p5 = is3D ? zero_f.
        where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, k_l, idx_y, idx_x})
        .unsqueeze(1)) : zero_f;
    T k_r = zero_l.where( (idx_z > d - 1 - bnd), idx_z + 1);
    T p6 = is3D ? zero_f.
        where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, k_r, idx_y, idx_x})
        .unsqueeze(1)) : zero_f;

    T neighborLeftObs = mCont.__and__(zeroBy.
         where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, idx_y, i_l}).eq(TypeObstacle)).unsqueeze(1));
    T neighborRightObs = mCont.__and__(zeroBy.
         where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, idx_y, i_r}).eq(TypeObstacle)).unsqueeze(1));
    T neighborBotObs = mCont.__and__(zeroBy.
         where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, j_l, idx_x}).eq(TypeObstacle)).unsqueeze(1));
    T neighborUpObs = mCont.__and__(zeroBy.
         where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, j_r, idx_x}).eq(TypeObstacle)).unsqueeze(1));
    T neighborBackObs = zeroBy;
    T neighborFrontObs = zeroBy;

    if (is3D) {
      T neighborBackObs = mCont.__and__(zeroBy.
           where(mCont.ne(1), flags.index({idx_b, zero_l, k_l, idx_y, idx_x}).eq(TypeObstacle)).unsqueeze(1));
      T neighborFrontObs = mCont.__and__(zeroBy.
           where(mCont.ne(1), flags.index({idx_b, zero_l, k_r, idx_y, idx_x}).eq(TypeObstacle)).unsqueeze(1));
    }

    p1.masked_scatter_(neighborLeftObs, pC.masked_select(neighborLeftObs));
    p2.masked_scatter_(neighborRightObs, pC.masked_select(neighborRightObs));
    p3.masked_scatter_(neighborBotObs, pC.masked_select(neighborBotObs));
    p4.masked_scatter_(neighborUpObs, pC.masked_select(neighborUpObs));
    p5.masked_scatter_(neighborBackObs, pC.masked_select(neighborBackObs));
    p6.masked_scatter_(neighborFrontObs, pC.masked_select(neighborFrontObs));

    const float denom = is3D ? 6 : 4;
    (*cur_p).masked_scatter_(mCont, ((p1 + p2 + p3 + p4 + p5 + p6 + div) / denom)
                                                             .masked_select(mCont));

    // Currrent iteration output is now in cur_pressure

    // Now calculate the change in pressure up to a sign (the sign might be 
    // incorrect, but we don't care).
    // p_delta = p - p_prev
    at::sub_out(p_delta, p, p_prev);
    p_delta.resize_({bsz, numel});
    // Calculate L2 norm over dim 2.
    at::norm_out(p_delta_norm, p_delta, at::Scalar(2), 1);
    p_delta.resize_({bsz, 1, d, h, w});
    residual = p_delta_norm.max();
    if (verbose) {
      std::cout << "Jacobi iteration " << (iter + 1) << ": residual "
                << residual << std::endl;
    }

    if (at::Scalar(residual).toFloat() < p_tol) {
      if (verbose) {
        std::cout << "Jacobi max residual fell below p_tol (" << p_tol
                  << ") (terminating)" << std::endl;
      }
      break;
    }

    iter++;
    if (iter >= max_iter) {
        if (verbose) {
          std::cout << "Jacobi max iteration count (" << max_iter
                    << ") reached (terminating)" << std::endl;
        }
        break;
    }

    // We haven't yet terminated.
    auto tmp = cur_p;
    cur_p = cur_p_prev;
    cur_p_prev = tmp;
  } // end while

  // If we terminated with the cur_pressure pointing to the tmp array, then we
  // have to copy the pressure back into the output tensor.
  if (cur_p == &p_prev) {
    p.copy_(p_prev);  // p = p_prev
  }

  // TODO: write mean-subtraction (FluidNet does it in Lua)
  return {p, residual};
}


} // namespace fluid

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("advect_scalar", &(fluid::advectScalar), "Advect Scalar");
    m.def("advect_vel", &(fluid::advectVel), "Advect Velocity");
    m.def("solve_linear_system", &(fluid::solveLinearSystemJacobi), "Solve Linear System using Jacobi's method");

}
