#pragma once

#include "ATen/ATen.h"
#include "grid/cell_type.h"

namespace fluid {

  typedef at::Tensor T;

// *****************************************************************************
// velocityUpdateForward
// *****************************************************************************

// Calculate the pressure gradient and subtract it into (i.e. calculate
// U' = U - grad(p)). Some care must be taken with handling boundary conditions.
// This function mimics correctVelocity in Manta.
// NOTE: velocity update is done IN-PLACE.
// 
// input U - vel field (size(2) can be 2 or 3, indicating 2D / 3D)
// input flags - input occupancy grid
// input pressure - scalar pressure field.

  void velocityUpdateForward(T& U, T& flags, T& pressure);

} // namespace fluid
