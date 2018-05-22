#pragma once

#include "grid/grid.h"

namespace fluid {

// *****************************************************************************
// velocityDivergenceForward
// *****************************************************************************

void velocityDivergenceForward(T& tensor_flags, T& tensor_u, 
       T& tensor_u_div, const bool is_3d);

} // namespace fluid
