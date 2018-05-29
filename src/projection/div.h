#pragma once

#include "grid/grid.h"

namespace fluid {

// *****************************************************************************
// velocityDivergenceForward
// *****************************************************************************

void velocityDivergenceForward(T& tensor_u, T& tensor_flags, 
       T& tensor_u_div);

} // namespace fluid
