#pragma once

#include "grid/grid.h"
#include "grid/cell_type.h"

namespace fluid {

// Solve the linear system using the Jacobi method.
// Note: Since we don't receive a velocity field, we need to receive the is3D
// flag from the caller.
// 
// input p: The output pressure field (i.e. the solution to A * p = div)
// input flags: The input flag grid.
// input div: The velocity divergence.
// input is3D: If true then we expect a 3D domain.
// input pTol: OPTIONAL (default = 1e-5), ||p - p_prev|| termination cond.
// input maxIter: OPTIONAL (default = 1000), max number of Jacobi iterations.
// input verbose: OPTIONAL (default = false), if true print out iteration res.
// 
// output: the max pTol across the batches.

float solveLinearSystemJacobi
(
   T& p,
   T& flags,
   T& div,
   const bool is_3d,
   const float p_tol,
   const int max_iter,
   const bool verbose
);

} // namespace fluid

