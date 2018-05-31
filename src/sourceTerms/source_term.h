#pragma once

#include "grid/grid.h"

namespace fluid {

// *****************************************************************************
// addBuoyancy
// *****************************************************************************

// Add buoyancy force. AddBuoyancy has a dt term.
// Note: Buoyancy is added IN-PLACE.
//
// @input U - vel field (size(2) can be 2 or 3, indicating 2D / 3D)
// @input flags - input occupancy grid
// @input density - scalar density grid.
// @input gravity - 3D vector indicating direction of gravity.
// @input dt - scalar timestep.

void addBuoyancy(T& tensor_u, T& tensor_flags, T& tensor_density,
   T& tensor_gravity, const float dt);

// *****************************************************************************
// addGravity
// *****************************************************************************

// Add gravity force. It has a dt term.
// Note: gravity is added IN-PLACE.
//
// @input U - vel field (size(2) can be 2 or 3, indicating 2D / 3D)
// @input flags - input occupancy grid
// @input gravity - 3D vector indicating direction of gravity.
// @input dt - scalar timestep.

void addGravity(T& tensor_u, T& tensor_flags, T& tensor_gravity,
   const float dt);

} // namespace fluid
