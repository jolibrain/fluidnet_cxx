#include "grid/grid.h"

namespace fluid {

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
// input p - scalar pressure field.

void velocityUpdateForward
(
    T& tensor_u,
    T& tensor_flags,
    T& tensor_p
);

} // namespace fluid
