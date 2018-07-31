#pragma once

#include <sstream>

#include "ATen/ATen.h"
#include "grid/grid.h"
#include "grid/cell_type.h"
#include "advect_type.h"
#include "calc_line_trace.h"

namespace fluid {

typedef at::Tensor T;

T SemiLagrangeEulerFluidNet
(
  T& flags, T& vel, T& src, T& maskBorder,
  float dt, float order_space,
  T& i, T& j, T& k,
  const bool line_trace,
  const bool sample_outside_fluid
);

T SemiLagrangeEulerFluidNetSavePos
(
  T& flags, T& vel, T& src, T& maskBorder,
  float dt, float order_space,
  T& i, T& j, T& k,
  const bool line_trace,
  const bool sample_outside_fluid,
  T& pos
);

T MacCormackCorrect
(
  T& flags, const T& old,
  const T& fwd, const T& bwd,
  const float strength,
  bool is_levelset
);

T getClampBounds
(
  const T& src, const T& pos, const T& flags,
  const bool sample_outside,
  T& clamp_min, T& clamp_max
);

T MacCormackClampFluidNet(
  T& flags, T& vel,
  const T& dst, const T& src,
  const T& fwd, const T& fwd_pos,
  const T& bwd_pos, const bool sample_outside
);

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
  const std::string method_str = "maccormackFluidNet",
  int bnd = 1,
  const bool sample_outside_fluid = false,
  const float maccormack_strength = 0.75
);

T SemiLagrangeEulerFluidNetMAC
(
  T& flags, T& vel, T& src, T& mask,
  float dt, float order_space,
  const bool line_trace,
  T& i, T& j, T& k
);

T MacCormackCorrectMAC
(
  T& flags, const T& old,
  const T& fwd, const T& bwd,
  const float strength,
  T& i, T& j, T& k
);

T doClampComponentMAC
(
  int chan,
  const T& flags, const T& dst,
  const T& orig,  const T& fwd,
  const T& pos, const T& vel
);

T MacCormackClampMAC
(
  const T& flags, const T& vel, const T& dval,
  const T& orig, const T& fwd, const T& mask,
  float dt,
  const T& i, const T& j, const T& k
);

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
  const std::string method_str = "maccormackFluidNet",
  int bnd = 1,
  const float maccormack_strength = 0.75
);

} // namespace fluid
