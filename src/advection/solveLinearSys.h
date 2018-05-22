#pragma once

#include "grid/grid.h"

namespace fluid {

void solveLinearSystemJacobi
(
   T* tensor_p,
   T* tensor_flags,
   T* tensor_div,
   T* tensor_p_prev,
   T* tensor_p_delta,
   T* tensor_p_delta_norm,
   const bool is_3d,
   const float p_tol,
   const int max_iter,
   const bool verbose
);

} // namespace fluid

