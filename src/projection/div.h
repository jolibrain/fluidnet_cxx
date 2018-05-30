#pragma once

#include "grid/grid.h"

namespace fluid {

// *****************************************************************************
// velocityDivergenceForward
// *****************************************************************************

// Calculate the velocity divergence (with boundary cond modifications). This is
// essentially a replica of makeRhs in Manta and FluidNet.
// 
// input U - input vel field (size(2) can be 2 or 3, indicating 2D / 3D)
// input flags - input occupancy grid
// input UDiv - output divergence (scalar field). 


void velocityDivergenceForward(T& tensor_u, T& tensor_flags, 
       T& tensor_u_div);

} // namespace fluid
