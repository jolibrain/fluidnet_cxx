#pragma once

#include "ATen/ATen.h"
#include "grid/cell_type.h"

namespace fluid {

  typedef at::Tensor T;

// *****************************************************************************
// velocityDivergenceForward
// *****************************************************************************

// Calculate the velocity divergence (with boundary cond modifications). This is
// essentially a replica of makeRhs in Manta and FluidNet.
// 
// input U - input vel field (size(2) can be 2 or 3, indicating 2D / 3D)
// input flags - input occupancy grid
// input UDiv - output divergence (scalar field). 

  void velocityDivergenceForward(T& U, T& flags, T& UDiv);

} // namespace fluid
