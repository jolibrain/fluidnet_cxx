#pragma once

#include "ATen/ATen.h"
#include "grid/grid.h"
#include "grid/cell_type.h"

namespace fluid {

typedef at::Tensor T;

// *****************************************************************************
// setWallBcsForward
// *****************************************************************************

// Enforce boundary conditions on velocity MAC Grid (i.e. set slip components).
// 
// @input U - input vel field (size(2) can be 2 or 3, indicating 2D / 3D)
// @input flags - input occupancy grid

void setWallBcsForward
(
    T& U,
    T& flags
);

} // namespace fluid
