#pragma once

#include "grid/grid.h"

namespace fluid {

// *****************************************************************************
// setWallBcsForward
// *****************************************************************************

// Enforce boundary conditions on velocity MAC Grid (i.e. set slip components).
// 
// @input U - input vel field (size(2) can be 2 or 3, indicating 2D / 3D)
// @input flags - input occupancy grid

void setWallBcsForward
(
    T& tensor_u,
    T& tensor_flags
);

} // namespace fluid
