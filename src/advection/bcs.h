#pragma once

#include "grid/grid.h"

namespace fluid {

// *****************************************************************************
// setWallBcsForward
// *****************************************************************************

void setWallBcsForward
(
    T& tensor_flags,
    T& tensor_u,
    const bool is_3d
);

} // namespace fluid
