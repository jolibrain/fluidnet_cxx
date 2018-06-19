#pragma once

#include "ATen/ATen.h"

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

void advectScalar
(
  float dt, T& src, T& U, T& flags, const std::string method_str, T& s_dst,
  const bool sample_outside_fluid, const float maccormack_strength, int bnd
); 

} // namespace fluid
