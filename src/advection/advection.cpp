#include "advection.h"

namespace fluid {
// *****************************************************************************
// Advect Scalar
// *****************************************************************************

T SemiLagrangeEulerFluidNet
(
    FlagGrid& flags,
    MACGrid& vel,
    RealGrid& src,
    float dt,
    int order_space,
    int32_t i, int32_t j, int32_t k, int32_t ibatch,
    const bool line_trace,
    const bool sample_outside_fluid
) {
  at::Backend bckd = flags.getBackend();
  at::ScalarType real = flags.getGridType();
  
  T b =  getType(bckd, at::kInt).scalarTensor(ibatch);
  if (!flags.isFluid(i, j, k, ibatch)) {
    // Don't advect solid geometry!
    return src(i, j, k, ibatch);
  }
  
  const T pos = getType(bckd, real).zeros({3});
  
  pos[0] = i + 0.5;
  pos[1] = j + 0.5;
  pos[2] = k + 0.5;
  
  T displacement = getType(bckd, real).zeros({3});
  displacement = vel.getCentered(i, j, k, ibatch) * (-dt);

  // Calculate a line trace from pos along displacement.
  // NOTE: this is expensive (MUCH more expensive than Manta's routines), but
  // can avoid some artifacts which would occur sampling into Geometry.
  T back_pos = getType(bckd, real).zeros({3});
  calcLineTrace(pos, displacement, flags, b, &back_pos, line_trace);

  // Finally, sample the field at this back position.
  if (!sample_outside_fluid) {
    return src.getInterpolatedWithFluidHi(flags, back_pos, order_space, b);
  } else {
    return src.getInterpolatedHi(back_pos, order_space, b);
  }
}

// This is the same kernel as the previous FluidNet Euler kernel, except it saves the
// particle trace position. This is used for FluidNet maccormack routine (it does
// a local search around these positions in clamp routine).
T SemiLagrangeEulerFluidNetSavePos
(
    FlagGrid& flags, 
    MACGrid& vel, 
    RealGrid& src,
    float dt,
    int order_space,
    int32_t i, int32_t j, int32_t k, int32_t ibatch,
    const bool line_trace,
    const bool sample_outside_fluid,
    VecGrid& pos
) {
  at::Backend bckd = flags.getBackend();
  at::ScalarType real = flags.getGridType();
  
  T b =  getType(bckd, at::kInt).scalarTensor(ibatch);

  const T start_pos = getType(bckd, real).zeros({3});

  start_pos[0] = i + 0.5;
  start_pos[1] = j + 0.5;
  start_pos[2] = k + 0.5;

  if (!flags.isFluid(i, j, k, ibatch)) {
    // Don't advect solid geometry!
    pos.set(i, j, k, ibatch, start_pos);
    return src(i, j, k, ibatch);
  }

  T displacement = getType(bckd, real).zeros({3});
  displacement = vel.getCentered(i, j, k, ibatch) * (-dt);

  // Calculate a line trace from pos along displacement.
  // NOTE: this is expensive (MUCH more expensive than Manta's routines), but
  // can avoid some artifacts which would occur sampling into Geometry.
  T back_pos = getType(bckd, real).zeros({3});
  calcLineTrace(start_pos, displacement, flags, b, &back_pos, line_trace);
  pos.set(i, j, k, ibatch, back_pos);

  // Finally, sample the field at this back position.
  if (!sample_outside_fluid) {
    return src.getInterpolatedWithFluidHi(flags, back_pos, order_space, b);
  } else {
    return src.getInterpolatedHi(back_pos, order_space, b);
  }
}

T MacCormackCorrect
(
    FlagGrid& flags,
    const RealGrid& old,
    const RealGrid& fwd,
    const RealGrid& bwd,
    const float strength,
    bool is_levelset,
    int32_t i, int32_t j, int32_t k, int32_t b
) {
  at::Backend bckd = flags.getBackend();
  at::ScalarType real = flags.getGridType();
  
  T dst = fwd(i, j, k, b);

  if (flags.isFluid(i, j, k, b)) {
    // Only correct inside fluid region.
    dst += strength * 0.5 * (old(i, j, k, b) - bwd(i, j, k, b));
  }
  return dst;
}

// FluidNet clamp routine. It is a search around a single input
// position for min and max values. If no valid values are found, then
// false is returned (indicating that a clamp shouldn't be performed) otherwise
// true is returned (and the clamp min and max bounds are set).
bool getClampBounds
(
    RealGrid src,
    T pos,
    const int32_t ibatch,
    FlagGrid flags,
    const bool sample_outside_fluid,
    T* clamp_min,
    T* clamp_max
) {
  at::Backend bckd = flags.getBackend();
  at::ScalarType real = flags.getGridType();

  T b =  getType(bckd, at::kInt).scalarTensor(ibatch);
 
  T minv = at::infer_type(pos).scalarTensor(std::numeric_limits<float>::infinity());
  T maxv = at::infer_type(pos).scalarTensor(-std::numeric_limits<float>::infinity());

  const T indx0 = getType(bckd, at::kInt).zeros({3});

  // clamp forward lookup to grid 
  indx0[0] = clamp(pos[0], 0, flags.xsize() - 1);
  indx0[1] = clamp(pos[1], 0, flags.ysize() - 1);
  indx0[2] =
    src.is_3d() ? clamp(pos[2], 0, flags.zsize() - 1)
    : getType(bckd, at::kInt).scalarTensor(0);

  // Some modification here. Instead of looking just to the RHS, we will search
  // all neighbors within a region.  This is more expensive but better handles
  // border cases.
  int32_t ncells = 0;
  for (T k = indx0[2] - 1; toBool(k <= indx0[2] + 1); k += 1) {
    for (T j = indx0[1] - 1; toBool(j <= indx0[1] + 1); j += 1) {
      for (T i = indx0[0] - 1; toBool(i <= indx0[0] + 1); i +=  1) {
        if (toBool((k < 0).__or__(k >= flags.zsize()).__or__
           (j < 0).__or__(j >= flags.ysize()).__or__
           (i < 0).__or__(i >= flags.xsize())) ) {
          // Outside bounds.
          continue;
        } else if (sample_outside_fluid || flags.isFluid(i, j, k, b)) {
          // Either we don't care about clamping to values inside the fluid, or
          // this is a fluid cell...
          getMinMax(minv, maxv, src(i, j, k, b));
          ncells++;
        }
      }
    }
  }

  if (ncells < 1) {
    // Only a single fluid cell found. Return false to indicate that a clamp
    // shouldn't be performed.
    return false;
  } else {
    *clamp_min = minv;
    *clamp_max = maxv;
    return true;
  }
}

T MacCormackClampFluidNet
(
    FlagGrid& flags,
    MACGrid& vel,
    const RealGrid& dst,
    const RealGrid& src,
    const RealGrid& fwd,
    float dt,
    const VecGrid& fwd_pos,
    const VecGrid& bwd_pos,
    const bool sample_outside_fluid,
    int32_t i, int32_t j, int32_t k, int32_t b
) {
  at::Backend bckd = flags.getBackend();
  at::ScalarType real = flags.getGridType();

   // Calculate the clamp bounds.
  T clamp_min = at::getType(bckd, real).scalarTensor(std::numeric_limits<float>::infinity());
  T clamp_max = at::getType(bckd, real).scalarTensor(-std::numeric_limits<float>::infinity());

  // Calculate the clamp bounds around the forward position.
  T pos = fwd_pos(i, j, k, b);
  const bool do_clamp_fwd = getClampBounds(
      src, pos, b, flags, sample_outside_fluid, &clamp_min, &clamp_max);

  // FluidNet: According to "An unconditionally stable maccormack method"
  // only a forward search is required.

  T dval;
  if (!do_clamp_fwd) {
    // If the cell is surrounded by fluid neighbors either in the fwd or
    // backward directions, then we need to revert to an euler step.
    dval = fwd(i, j, k, b);
  } else {
    // We found valid values with which to clamp the maccormack corrected
    // quantity. Apply this clamp.
    dval = clamp(dst(i, j, k, b), at::Scalar(clamp_min), at::Scalar(clamp_max));
  }

  return dval;
}

void advectScalar
(
    float dt,
    T& tensor_flags,
    T& tensor_u,
    T& tensor_s,
    T& tensor_s_dst,
    T& tensor_fwd,
    T& tensor_bwd,
    T& tensor_fwd_pos,
    T& tensor_bwd_pos,
    const bool is_3d,
    const std::string method_str,
    const int32_t boundary_width,
    const bool sample_outside_fluid,
    const float maccormack_strength
) {
  // TODO: check sizes (see lua). We also treat 2D advection as 3D (with depth = 1) and
  // no 'w' component for velocity.
  FlagGrid flags(tensor_flags, is_3d);
  MACGrid vel(tensor_u, is_3d);
  RealGrid src(tensor_s, is_3d);
  RealGrid dst(tensor_s_dst, is_3d);

  // The maccormack method also needs fwd and bwd temporary arrays.
  RealGrid fwd(tensor_fwd, is_3d);
  RealGrid bwd(tensor_bwd, is_3d); 
  VecGrid fwd_pos(tensor_fwd_pos, is_3d);
  VecGrid bwd_pos(tensor_bwd_pos, is_3d);

  AdvectMethod method = StringToAdvectMethod(method_str);

  const bool is_levelset = false;  // We never advect them.
  const int order_space = 1;
  // A full line trace along every ray is expensive but correct (applies to our
  // methods only).
  const bool line_trace = true;

  const int32_t nbatch = flags.nbatch();
  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();

  at::Backend bckd = flags.getBackend();
  at::ScalarType real = flags.getGridType();
  const T posCorrected = getType(bckd, real).zeros({3});

  for (int32_t b = 0; b < nbatch; b++) {
    const int32_t bnd = boundary_width;
    int32_t k, j, i;
    RealGrid* cur_dst = (method == ADVECT_MACCORMACK_FLUIDNET) ? &fwd : &dst;

    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
           
            if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Manta zeros stuff on the border.
            (*cur_dst)(i, j, k, b) = 0;
            posCorrected[0] = i + 0.5;
            posCorrected[1] = j + 0.5;
            posCorrected[2] = k + 0.5;
            fwd_pos.set(i, j, k, b, posCorrected);
            continue;
          }

          // Forward step.
          T val;
          if (method == ADVECT_EULER_FLUIDNET) {
            val = SemiLagrangeEulerFluidNet(
                flags, vel, src, dt, order_space, i, j, k, b, line_trace,
                sample_outside_fluid);
          } else if (method == ADVECT_MACCORMACK_FLUIDNET) {
            val = SemiLagrangeEulerFluidNetSavePos(
                flags, vel, src, dt, order_space, i, j, k, b, line_trace,
                sample_outside_fluid, fwd_pos);
          } else {
            AT_ERROR("Advection method not supported!");
          }

          (*cur_dst)(i, j, k, b) = val;
        }
      }
    }

    if (method != ADVECT_MACCORMACK_FLUIDNET) {
      // We're done. The forward Euler step is already in the output array.
    } else {
      // Otherwise we need to do the backwards step (which is a SemiLagrange
      // step on the forward data - hence we needed to finish the above loops
      // beforemoving on).
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) {
            if (i < bnd || i > xsize - 1 - bnd ||
                j < bnd || j > ysize - 1 - bnd ||
                (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
              bwd(i, j, k, b) = 0;
              posCorrected[0] = i + 0.5;
              posCorrected[1] = j + 0.5;
              posCorrected[2] = k + 0.5;
              bwd_pos.set(i, j, k, b, posCorrected);
              continue; 
            } 

            // Backwards step.
            if (method == ADVECT_MACCORMACK_FLUIDNET) {
              bwd(i, j, k, b) = SemiLagrangeEulerFluidNetSavePos(
                  flags, vel, fwd, -dt,  order_space, i, j, k, b, line_trace,
                  sample_outside_fluid, bwd_pos);
            }
          }
        }
      }

      // Now compute the correction.
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) { 
            dst(i, j, k, b) = MacCormackCorrect(
                flags, src, fwd, bwd, maccormack_strength, is_levelset,
                i, j, k, b);
          }
        }
      }

      // Now perform clamping.
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) {
            if (i < bnd || i > xsize - 1 - bnd ||
                j < bnd || j > ysize - 1 - bnd ||
                (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
              continue;
            }
            if (method == ADVECT_MACCORMACK_FLUIDNET) {
              dst(i, j, k, b) = MacCormackClampFluidNet(
                  flags, vel, dst, src, fwd, dt, fwd_pos, bwd_pos,
                  sample_outside_fluid, i, j, k, b);  
            }
          }
        }
      }
    }
  }

}


// *****************************************************************************
// Advect Velocity
// *****************************************************************************

T SemiLagrangeEulerFluidNetMAC(
    FlagGrid& flags,
    MACGrid& vel,
    MACGrid& src,
    float dt,
    int order_space,
    const bool line_trace,
    int32_t i, int32_t j, int32_t k, int32_t b) {
  if (!flags.isFluid(i, j, k, b)) {
    // Don't advect solid geometry!
    return src(i, j, k, b);
  }
  at::Backend bckd = flags.getBackend();
  at::ScalarType real = flags.getGridType();

  T ibatch =  getType(bckd, at::kInt).scalarTensor(b);
  T nchan = getType(bckd, at::kInt).arange(3);

  // Get correct velocity at MAC position.
  // No need to shift xpos etc. as lookup field is also shifted.

  const T pos = getType(bckd, real).zeros({3});

  pos[0] = i + 0.5;
  pos[1] = j + 0.5;
  pos[2] = k + 0.5;


  // FluidNet: We floatly want to clamp to the SMALLEST of the steps in each
  // dimension, however this is OK for now (because doing so would expensive)...
  T xpos = getType(bckd, real).zeros({3});
  calcLineTrace(pos, vel.getAtMACX(i, j, k, b) * (-dt), flags, ibatch, &xpos,
                line_trace);
  const T vx = src.getInterpolatedComponentHi(xpos, order_space, nchan[0], ibatch);

  T ypos = getType(bckd, real).zeros({3});
  calcLineTrace(pos, vel.getAtMACY(i, j, k, b) * (-dt), flags, ibatch, &ypos,
                line_trace);
  const T vy = src.getInterpolatedComponentHi(ypos, order_space, nchan[1], ibatch);

  T vz = getType(bckd, real).scalarTensor(0);
  if (vel.is_3d()) {
    T zpos = getType(bckd, real).zeros({3});
    calcLineTrace(pos, vel.getAtMACZ(i, j, k, b) * (-dt), flags, ibatch, &zpos,
                  line_trace);
    vz = src.getInterpolatedComponentHi(zpos, order_space, nchan[2], ibatch);
  } else {
    // vz = 0 (already at initialization)
  }
  
  T ret = getType(bckd, real).zeros({3});
  ret[0] = vx;
  ret[1] = vy;
  ret[2] = vz;

  return ret;
}

T SemiLagrangeMAC(
    FlagGrid& flags,
    MACGrid& vel,
    MACGrid& src,
    float dt,
    int order_space,
    int32_t i, int32_t j, int32_t k, int32_t b) {
  at::Backend bckd = flags.getBackend();
  at::ScalarType real = flags.getGridType();

  T ibatch =  getType(bckd, at::kInt).scalarTensor(b);
  T nchan = getType(bckd, at::kInt).arange(3);
  
  // Get correct velocity at MAC position.
  // No need to shift xpos etc. as lookup field is also shifted.

  const T pos = getType(bckd, real).zeros({3});

  pos[0] = i + 0.5;
  pos[1] = j + 0.5;
  pos[2] = k + 0.5;

  T xpos = pos - vel.getAtMACX(i, j, k, b) * dt;
  const T vx = src.getInterpolatedComponentHi(xpos, order_space, nchan[0], ibatch);

  T ypos = pos - vel.getAtMACY(i, j, k, b) * dt;
  const T vy = src.getInterpolatedComponentHi(ypos, order_space, nchan[1], ibatch);

  T vz = getType(bckd, real).scalarTensor(0);
  if (vel.is_3d()) {
    T zpos = pos - vel.getAtMACZ(i, j, k, b) * dt;
    vz = src.getInterpolatedComponentHi(zpos, order_space, nchan, ibatch);
  } else {
    // vz = 0 (already at initialization);
  }
  
  T ret = getType(bckd, real).zeros({3});
  ret[0] = vx;
  ret[1] = vy;
  ret[2] = vz;

  return ret;
}

T MacCormackCorrectMAC(
    FlagGrid& flags,
    const MACGrid& old,
    const MACGrid& fwd,
    const MACGrid& bwd,
    const float strength,
    int32_t i, int32_t j, int32_t k, int32_t b) {
  at::Backend bckd = flags.getBackend();
  at::ScalarType real = flags.getGridType();

  bool skip[3] = {false, false, false};

  if (!flags.isFluid(i, j, k, b)) {
    skip[0] = true;
    skip[1] = true;
    skip[2] = true;
  }

  // Note: in Manta code there's a isMAC boolean that is always true.
  if ((i > 0) && (!flags.isFluid(i - 1, j, k, b))) {
    skip[0] = true;
  }
  if ((j > 0) && (!flags.isFluid(i, j - 1, k, b))) {
    skip[1] = true;
  }
  if (flags.is_3d()) {
    if ((k > 0) && (!flags.isFluid(i, j, k - 1, b))) {
      skip[2] = true;
    }
  }

  T dst = getType(bckd, real).zeros({3});

  const int32_t dim = flags.is_3d() ? 3 : 2;
  for (int32_t c = 0; c < dim; ++c) {
    if (skip[c]) {
      dst[c] = fwd(i, j, k, c, b);
    } else {
      // perform actual correction with given strength.
      dst[c] = fwd(i, j, k, c, b) + strength * 0.5 * (old(i, j, k, c, b) -
                                                      bwd(i, j, k, c, b));
    }
  }

  return dst;
}

template <int32_t chan>
T doClampComponentMAC(
    const T& gridSize,
    T dst,
    const MACGrid& orig,
    T fwd, 
    const T& pos, const T& vel,
    int32_t b) {
  at::Backend bckd = orig.getBackend();
  at::ScalarType real = orig.getGridType();
  
  T c = getType(bckd, at::kInt).scalarTensor(chan);
  T ibatch =  getType(bckd, at::kInt).scalarTensor(b);

  T minv = at::infer_type(pos).scalarTensor(std::numeric_limits<float>::infinity());
  T maxv = at::infer_type(pos).scalarTensor(-std::numeric_limits<float>::infinity());

  // forward (and optionally) backward
  T positions[2] = getType(bckd, at::kInt).zeros({2,3});
  positions[0] = (pos - vel).toType(getType(bckd, at::kInt));
  positions[1] = (pos + vel).toType(getType(bckd, at::kInt));

  for (int32_t l = 0; l < 2; ++l) {
    const T& curr_pos = positions[l];
    const T indx0 = getType(bckd, at::kInt).zeros({3});
    const T indx1 = getType(bckd, at::kInt).zeros({3});

    // clamp forward lookup to grid 
    indx0[0] = clamp(curr_pos[0], 0, at::Scalar(gridSize[0] - 1));
    indx0[1] = clamp(curr_pos[1], 0, at::Scalar(gridSize[1] - 1));
    indx0[2] = clamp(curr_pos[2], 0,
                             (orig.is_3d() ? at::Scalar(gridSize[2] - 1) : at::Scalar(1)));
    indx1[0] = indx0[0] + 1;
    indx1[1] = indx0[1] + 1;
    indx1[2] = (orig.is_3d() ? (indx0[2] + 1) : indx0[2]);
    if (!orig.isInBounds(indx0, 0) ||
        !orig.isInBounds(indx1, 0)) {
      return fwd;
    }

    // find min/max around source pos
    getMinMax(minv, maxv, orig(indx0[0], indx0[1], indx0[2], c, ibatch));
    getMinMax(minv, maxv, orig(indx1[0], indx0[1], indx0[2], c, ibatch));
    getMinMax(minv, maxv, orig(indx0[0], indx1[1], indx0[2], c, ibatch));
    getMinMax(minv, maxv, orig(indx1[0], indx1[1], indx0[2], c, ibatch));

    if (orig.is_3d()) {
      getMinMax(minv, maxv, orig(indx0[0], indx0[1], indx1[2], c, ibatch));
      getMinMax(minv, maxv, orig(indx1[0], indx0[1], indx1[2], c, ibatch));
      getMinMax(minv, maxv, orig(indx0[0], indx1[1], indx1[2], c, ibatch));
      getMinMax(minv, maxv, orig(indx1[0], indx1[1], indx1[2], c, ibatch));
    }
  }

  dst = clamp(dst, at::Scalar(minv), at::Scalar(maxv));
  return dst;
}

T MacCormackClampMAC(
    FlagGrid& flags,
    MACGrid& vel,
    T dval,
    const MACGrid& orig,
    const MACGrid& fwd,
    float dt,
    int32_t i, int32_t j, int32_t k, int32_t b) {
  at::Backend bckd = flags.getBackend();
  at::ScalarType real = flags.getGridType();

  T nchan = getType(bckd, at::kInt).arange(3);
  // Get correct velocity at MAC position.
  // No need to shift xpos etc. as lookup field is also shifted.

  const T pos = getType(bckd, real).zeros({3});

  pos[0] = i;
  pos[1] = j;
  pos[2] = k;
  
  T dfwd = fwd(i, j, k, b);
  T gridUpper = flags.getSize() - 1;

  dval[0] = doClampComponentMAC<0>(gridUpper, dval[0], orig, dfwd[0], pos,
                                  vel.getAtMACX(i, j, k, b) * dt, b);
  
  dval[1] = doClampComponentMAC<1>(gridUpper, dval[1], orig, dfwd[1], pos,
                                  vel.getAtMACY(i, j, k, b) * dt, b);
  if (flags.is_3d()) {
    dval[2] = doClampComponentMAC<2>(gridUpper, dval[2], orig, dfwd[2], pos,
                                    vel.getAtMACZ(i, j, k, b) * dt, b);
  } else {
    dval[2] = 0;
  }

  // Note (from Manta): The MAC version currently does not check whether source 
  // points were inside an obstacle! (unlike centered version) this would need
  // to be done for each face separately to stay symmetric.
  
  return dval;
}

void advectVel
(
    float dt,
    T& tensor_flags,
    T& tensor_u,
    T& tensor_u_dst,
    T& tensor_fwd,
    T& tensor_bwd,
    const bool is_3d,
    const std::string method_str,
    const int32_t boundary_width,
    const float maccormack_strength
) {
  // TODO: Check sizes. We also treat 2D advection as 3D (with depth = 1) and
  // no 'w' component for velocity.

  AdvectMethod method = StringToAdvectMethod(method_str);

  const int order_space = 1;
  // A full line trace along every ray is expensive but correct (applies to FluidNet
  // methods only).
  const bool line_trace = true;

  FlagGrid flags(tensor_flags, is_3d);
  MACGrid vel(tensor_u, is_3d);

  // We always do self-advection, but we could point orig to another tensor.
  MACGrid orig(tensor_u, is_3d);
  MACGrid dst(tensor_u_dst, is_3d);

  // The maccormack method also needs fwd and bwd temporary arrays.
  MACGrid fwd(tensor_fwd, is_3d);
  MACGrid bwd(tensor_bwd, is_3d);

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();

  at::Backend bckd = flags.getBackend();
  at::ScalarType real = flags.getGridType();
  const T posCorrected = getType(bckd, real).zeros({3});

  for (int32_t b = 0; b < nbatch; b++) {
    MACGrid* cur_dst = (method == ADVECT_MACCORMACK_FLUIDNET) ?
                                  &fwd : &dst;
    int32_t k, j, i;
    const int32_t bnd = 1;
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Manta zeros stuff on the border.
            cur_dst->setSafe(i, j, k, b, posCorrected);
            continue;
          }

          // Forward step.
          T val;
          if (method == ADVECT_EULER_FLUIDNET ||
              method == ADVECT_MACCORMACK_FLUIDNET) {
            val = SemiLagrangeEulerFluidNetMAC(
                flags, vel, orig, dt, order_space, line_trace, i, j, k, b);
          }
          else {
          std::cout << "No defined method for MAC advection" << std::endl;
          }
          cur_dst->setSafe(i, j, k, b, val);  // Store in the output array
        }
      }
    }

    if (method != ADVECT_MACCORMACK_FLUIDNET) {
      // We're done. The forward Euler step is already in the output array.
    } else {
      // Otherwise we need to do the backwards step (which is a SemiLagrange
      // step on the forward data - hence we needed to finish the above loops
      // before moving on).
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) {
            if (i < bnd || i > xsize - 1 - bnd ||
                j < bnd || j > ysize - 1 - bnd ||
                (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
              bwd.setSafe(i, j, k, b, posCorrected);
              continue; 
            } 

            // Backwards step.
            if (method == ADVECT_MACCORMACK_FLUIDNET) {
              bwd.setSafe(i, j, k, b, SemiLagrangeEulerFluidNetMAC(
                  flags, vel, fwd, -dt, order_space, line_trace, i, j, k, b));
            }
          }
        }
      }

      // Now compute the correction.
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) { 
            dst.setSafe(i, j, k, b, MacCormackCorrectMAC(
                flags, orig, fwd, bwd, maccormack_strength, i, j, k, b));
          }
        }
      }
      
      // Now perform clamping.
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) {
            if (i < bnd || i > xsize - 1 - bnd ||
                j < bnd || j > ysize - 1 - bnd ||
                (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
              continue;
            }
            // TODO(tompson): Perform our own clamping.
            const T dval = dst(i, j, k, b);
            dst.setSafe(i, j, k, b, MacCormackClampMAC(
                flags, vel, dval, orig, fwd, dt, i, j, k, b));
          }
        }
      }
    }
  }

}

} // namespace fluid
