#include "calc_line_trace.h"
#include "advect_type.h"

// *****************************************************************************
// advectScalar
// *****************************************************************************

float SemiLagrangeEulerFluidNet
(
    FlagGrid& flags,
    MACGrid& vel,
    RealGrid& src,
    float dt,
    int order_space,
    int32_t i, int32_t j, int32_t k, int32_t b,
    const bool line_trace,
    const bool sample_outside_fluid
) {
  if (!flags.isFluid(i, j, k, b)) {
    // Don't advect solid geometry!
    return src(i, j, k, b);
  }

  const vec3 pos = vec3((float)i + (float)0.5,
                        (float)j + (float)0.5,
                        (float)k + (float)0.5);
  vec3 displacement = vel.getCentered(i, j, k, b) * (-dt);

  // Calculate a line trace from pos along displacement.
  // NOTE: this is expensive (MUCH more expensive than Manta's routines), but
  // can avoid some artifacts which would occur sampling into Geometry.
  vec3 back_pos;
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
float SemiLagrangeEulerFluidNetSavePos
(
    FlagGrid& flags, 
    MACGrid& vel, 
    RealGrid& src,
    float dt,
    int order_space,
    int32_t i, int32_t j, int32_t k, int32_t b,
    const bool line_trace,
    const bool sample_outside_fluid,
    VecGrid& pos
) {
  if (!flags.isFluid(i, j, k, b)) {
    // Don't advect solid geometry!
    pos.set(i, j, k, b, vec3(i, j, k) + (float)0.5);
    return src(i, j, k, b);
  }

  const vec3 start_pos = vec3((float)i + (float)0.5,
                                                  (float)j + (float)0.5,
                                                  (float)k + (float)0.5);
  vec3 displacement = vel.getCentered(i, j, k, b) * (-dt);

  // Calculate a line trace from pos along displacement.
  // NOTE: this is expensive (MUCH more expensive than Manta's routines), but
  // can avoid some artifacts which would occur sampling into Geometry.
  vec3 back_pos;
  calcLineTrace(start_pos, displacement, flags, b, &back_pos, line_trace);
  pos.set(i, j, k, b, back_pos);

  // Finally, sample the field at this back position.
  if (!sample_outside_fluid) {
    return src.getInterpolatedWithFluidHi(flags, back_pos, order_space, b);
  } else {
    return src.getInterpolatedHi(back_pos, order_space, b);
  }
}

float MacCormackCorrect
(
    FlagGrid& flags,
    const RealGrid& old,
    const RealGrid& fwd,
    const RealGrid& bwd,
    const float strength,
    bool is_levelset,
    int32_t i, int32_t j, int32_t k, int32_t b
) {
  float dst = fwd(i, j, k, b);

  if (flags.isFluid(i, j, k, b)) {
    // Only correct inside fluid region.
    dst += strength * 0.5 * (old(i, j, k, b) - bwd(i, j, k, b));
  }
  return dst;
}

void getMinMax(float& minv, float& maxv, const float& val) {
  if (val < minv) {
    minv = val;
  }
  if (val > maxv) {
    maxv = val;
  }
}

float clamp(const float val, const float min, const float max) {
  return std::min<float>(max, std::max<float>(min, val));
}

// FluidNet clamp routine. It is a search around a single input
// position for min and max values. If no valid values are found, then
// false is returned (indicating that a clamp shouldn't be performed) otherwise
// true is returned (and the clamp min and max bounds are set).
static float getClampBounds
(
    RealGrid src,
    vec3 pos,
    const int32_t b,
    FlagGrid flags,
    const bool sample_outside_fluid,
    float* clamp_min,
    float* clamp_max
) {
  float minv = std::numeric_limits<float>::infinity();
  float maxv = -std::numeric_limits<float>::infinity();

  // clamp forward lookup to grid 
  const int32_t i0 = clamp((int32_t)pos.x, 0, flags.xsize() - 1);
  const int32_t j0 = clamp((int32_t)pos.y, 0, flags.ysize() - 1);
  const int32_t k0 =
    src.is_3d() ? clamp((int32_t)pos.z, 0, flags.zsize() - 1) : 0;
  // Some modification here. Instead of looking just to the RHS, we will search
  // all neighbors within a region.  This is more expensive but better handles
  // border cases.
  int32_t ncells = 0;
  for (int32_t k = k0 - 1; k <= k0 + 1; k++) {
    for (int32_t j = j0 - 1; j <= j0 + 1; j++) {
      for (int32_t i = i0 - 1; i <= i0 + 1; i++) {
        if (k < 0 || k >= flags.zsize() ||
            j < 0 || j >= flags.ysize() ||
            i < 0 || i >= flags.xsize()) {
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

float MacCormackClampFluidNet
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

   // Calculate the clamp bounds.
  float clamp_min = std::numeric_limits<float>::infinity();
  float clamp_max = -std::numeric_limits<float>::infinity();

  // Calculate the clamp bounds around the forward position.
  vec3 pos = fwd_pos(i, j, k, b);
  const bool do_clamp_fwd = getClampBounds(
      src, pos, b, flags, sample_outside_fluid, &clamp_min, &clamp_max);

  // FluidNet: According to "An unconditionally stable maccormack method"
  // only a forward search is required.

  float dval;
  if (!do_clamp_fwd) {
    // If the cell is surrounded by fluid neighbors either in the fwd or
    // backward directions, then we need to revert to an euler step.
    dval = fwd(i, j, k, b);
  } else {
    // We found valid values with which to clamp the maccormack corrected
    // quantity. Apply this clamp.
    dval = clamp(dst(i, j, k, b), clamp_min, clamp_max);
  }

  return dval;
}

void advectScalar
(
    float dt,
    at::Tensor* tensor_flags,
    at::Tensor* tensor_u,
    at::Tensor* tensor_s,
    at::Tensor* tensor_s_dst,
    at::Tensor* tensor_fwd,
    at::Tensor* tensor_bwd,
    at::Tensor* fwd_pos,
    at::Tensor* bwd_pos,
    const bool is_3d,
    const std::string method_str
    const int32_t boundary_width
    const bool sample_outside_fluid
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

  AdvectMethod method = StringToAdvectMethod(L, method_str);

  const bool is_levelset = false;  // We never advect them.
  const int order_space = 1;
  // A full line trace along every ray is expensive but correct (applies to our
  // methods only).
  const bool line_trace = true;

  const int32_t nbatch = flags.nbatch();
  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();

  for (int32_t b = 0; b < nbatch; b++) {
    const int32_t bnd = 1;
    int32_t k, j, i;
    RealGrid* cur_dst = (method == ADVECT_MACCORMACK_FLUIDNET) ?
                                   &fwd : &dst;

#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Manta zeros stuff on the border.
            (*cur_dst)(i, j, k, b) = 0;
            fwd_pos.set(i, j, k, b, vec3(i, j, k) + (float)0.5);
            continue;
          }

          // Forward step.
          float val;
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
#pragma omp parallel for collapse(3) private(k, j, i)
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) {
            if (i < bnd || i > xsize - 1 - bnd ||
                j < bnd || j > ysize - 1 - bnd ||
                (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
              bwd(i, j, k, b) = 0;
              bwd_pos.set(i, j, k, b, vec3(i, j, k) + (float)0.5);
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
#pragma omp parallel for collapse(3) private(k, j, i)
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
#pragma omp parallel for collapse(3) private(k, j, i)
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
// advectVel
// *****************************************************************************

vec3 SemiLagrangeEulerFluidNetMAC(
    FlagGrid& flags, MACGrid& vel, MACGrid& src,
    float dt, int order_space, const bool line_trace, int32_t i, int32_t j,
    int32_t k, int32_t b) {
  if (!flags.isFluid(i, j, k, b)) {
    // Don't advect solid geometry!
    return src(i, j, k, b);
  }

  // Get correct velocity at MAC position.
  // No need to shift xpos etc. as lookup field is also shifted.
  const vec3 pos(static_cast<float>(i) + 0.5,
                           static_cast<float>(j) + 0.5,
                           static_cast<float>(k) + 0.5);

  // FluidNet: We floatly want to clamp to the SMALLEST of the steps in each
  // dimension, however this is OK for now (because doing so would expensive)...
  vec3 xpos;
  calcLineTrace(pos, vel.getAtMACX(i, j, k, b) * (-dt), flags, b, &xpos,
                line_trace);
  const float vx = src.getInterpolatedComponentHi(xpos, order_space, 0, b);

  vec3 ypos;
  calcLineTrace(pos, vel.getAtMACY(i, j, k, b) * (-dt), flags, b, &ypos,
                line_trace);
  const float vy = src.getInterpolatedComponentHi(ypos, order_space, 1, b);

  float vz;
  if (vel.is_3d()) {
    vec3 zpos;
    calcLineTrace(pos, vel.getAtMACZ(i, j, k, b) * (-dt), flags, b, &zpos,
                  line_trace);
    vz = src.getInterpolatedComponentHi(zpos, order_space, 2, b);
  } else {
    vz = 0;
  }

  return vec3(vx, vy, vz);
}

vec3 SemiLagrangeMAC(
    FlagGrid& flags, MACGrid& vel, MACGrid& src,
    float dt, int order_space, int32_t i, int32_t j, int32_t k, int32_t b) {
  // Get correct velocity at MAC position.
  // No need to shift xpos etc. as lookup field is also shifted.
  const vec3 pos(static_cast<float>(i) + 0.5,
                           static_cast<float>(j) + 0.5,
                           static_cast<float>(k) + 0.5);

  vec3 xpos = pos - vel.getAtMACX(i, j, k, b) * dt;
  const float vx = src.getInterpolatedComponentHi(xpos, order_space, 0, b);

  vec3 ypos = pos - vel.getAtMACY(i, j, k, b) * dt;
  const float vy = src.getInterpolatedComponentHi(ypos, order_space, 1, b);

  float vz;
  if (vel.is_3d()) {
    vec3 zpos = pos - vel.getAtMACZ(i, j, k, b) * dt;
    vz = src.getInterpolatedComponentHi(zpos, order_space, 2, b);
  } else {
    vz = 0;
  }

  return vec3(vx, vy, vz);
}

vec3 MacCormackCorrectMAC(
    FlagGrid& flags, const MACGrid& old,
    const MACGrid& fwd, const MACGrid& bwd,
    const float strength, int32_t i, int32_t j, int32_t k, int32_t b) {
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

  vec3 dst(0, 0, 0);

  const int32_t dim = flags.is_3d() ? 3 : 2;
  for (int32_t c = 0; c < dim; ++c) {
    if (skip[c]) {
      dst(c) = fwd(i, j, k, c, b);
    } else {
      // perform actual correction with given strength.
      dst(c) = fwd(i, j, k, c, b) + strength * 0.5 * (old(i, j, k, c, b) -
                                                      bwd(i, j, k, c, b));
    }
  }

  return dst;
}

template <int32_t c>
float doClampComponentMAC(
    const Int3& gridSize, float dst, const MACGrid& orig,
    float fwd, const vec3& pos, const vec3& vel,
    int32_t b) {
  float minv = std::numeric_limits<float>::max();
  float maxv = -std::numeric_limits<float>::max();

  // forward (and optionally) backward
  Int3 positions[2];
  positions[0] = toInt3(pos - vel);
  positions[1] = toInt3(pos + vel);

  for (int32_t l = 0; l < 2; ++l) {
    Int3& curr_pos = positions[l];

    // clamp forward lookup to grid 
    const int32_t i0 = clamp(curr_pos.x, 0, gridSize.x - 1);
    const int32_t j0 = clamp(curr_pos.y, 0, gridSize.y - 1);
    const int32_t k0 = clamp(curr_pos.z, 0,
                             (orig.is_3d() ? (gridSize.z - 1) : 1));
    const int32_t i1 = i0 + 1;
    const int32_t j1 = j0 + 1;
    const int32_t k1 = (orig.is_3d() ? (k0 + 1) : k0);
    if (!orig.isInBounds(Int3(i0, j0, k0), 0) ||
        !orig.isInBounds(Int3(i1, j1, k1), 0)) {
      return fwd;
    }

    // find min/max around source pos
    getMinMax(minv, maxv, orig(i0, j0, k0, c, b));
    getMinMax(minv, maxv, orig(i1, j0, k0, c, b));
    getMinMax(minv, maxv, orig(i0, j1, k0, c, b));
    getMinMax(minv, maxv, orig(i1, j1, k0, c, b));

    if (orig.is_3d()) {
      getMinMax(minv, maxv, orig(i0, j0, k1, c, b));
      getMinMax(minv, maxv, orig(i1, j0, k1, c, b));
      getMinMax(minv, maxv, orig(i0, j1, k1, c, b));
      getMinMax(minv, maxv, orig(i1, j1, k1, c, b));
    }
  }

  dst = clamp(dst, minv, maxv);
  return dst;
}

vec3 MacCormackClampMAC(
    FlagGrid& flags, MACGrid& vel,
    vec3 dval, const MACGrid& orig,
    const MACGrid& fwd, float dt,
    int32_t i, int32_t j, int32_t k, int32_t b) {
  vec3 pos(static_cast<float>(i), static_cast<float>(j),
                     static_cast<float>(k));
  vec3 dfwd = fwd(i, j, k, b);
  Int3 gridUpper = flags.getSize() - 1;

  dval.x = doClampComponentMAC<0>(gridUpper, dval.x, orig, dfwd.x, pos,
                                  vel.getAtMACX(i, j, k, b) * dt, b);
  dval.y = doClampComponentMAC<1>(gridUpper, dval.y, orig, dfwd.y, pos,
                                  vel.getAtMACY(i, j, k, b) * dt, b);
  if (flags.is_3d()) {
    dval.z = doClampComponentMAC<2>(gridUpper, dval.z, orig, dfwd.z, pos,
                                    vel.getAtMACZ(i, j, k, b) * dt, b);
  } else {
    dval.z = 0;
  }

  // Note (from Manta): The MAC version currently does not check whether source 
  // points were inside an obstacle! (unlike centered version) this would need
  // to be done for each face separately to stay symmetric.
  
  return dval;
}

void advectVel
(
    float dt,
    at::Tensor* tensor_flags,
    at::Tensor* tensor_u,
    at::Tensor* tensor_u_dst,
    at::Tensor* tensor_fwd,
    at::Tensor* tensor_bwd,
    const bool is_3d,
    const std::string method_str
    const int32_t boundary_width
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

  for (int32_t b = 0; b < nbatch; b++) {
    MACGrid* cur_dst = (method == ADVECT_MACCORMACK_FLUIDNET) ?
                                  &fwd : &dst;
    int32_t k, j, i;
    const int32_t bnd = 1;
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Manta zeros stuff on the border.
            cur_dst->setSafe(i, j, k, b, vec3(0, 0, 0));
            continue;
          }

          // Forward step.
          vec3 val;
          if (method == ADVECT_EULER_FLUIDNET ||
              method == ADVECT_MACCORMACK_FLUIDNET) {
            val = SemiLagrangeEulerOursMAC(
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
#pragma omp parallel for collapse(3) private(k, j, i)
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) {
            if (i < bnd || i > xsize - 1 - bnd ||
                j < bnd || j > ysize - 1 - bnd ||
                (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
              bwd.setSafe(i, j, k, b, vec3(0, 0, 0));
              continue; 
            } 

            // Backwards step.
            if (method == ADVECT_MACCORMACK_FLUIDNET) {
              bwd.setSafe(i, j, k, b, SemiLagrangeEulerOursMAC(
                  flags, vel, fwd, -dt, order_space, line_trace, i, j, k, b));
            }
          }
        }
      }

      // Now compute the correction.
#pragma omp parallel for collapse(3) private(k, j, i)
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) { 
            dst.setSafe(i, j, k, b, MacCormackCorrectMAC(
                flags, orig, fwd, bwd, maccormack_strength, i, j, k, b));
          }
        }
      }
      
      // Now perform clamping.
#pragma omp parallel for collapse(3) private(k, j, i)
      for (k = 0; k < zsize; k++) { 
        for (j = 0; j < ysize; j++) { 
          for (i = 0; i < xsize; i++) {
            if (i < bnd || i > xsize - 1 - bnd ||
                j < bnd || j > ysize - 1 - bnd ||
                (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
              continue;
            }
            // TODO(tompson): Perform our own clamping.
            const vec3 dval = dst(i, j, k, b);
            dst.setSafe(i, j, k, b, MacCormackClampMAC(
                flags, vel, dval, orig, fwd, dt, i, j, k, b));
          }
        }
      }
    }
  }

}
