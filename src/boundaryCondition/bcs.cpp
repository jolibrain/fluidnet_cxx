#include "bcs.h"

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
) {
  // Check arguments.
  AT_ASSERT(tensor_u.dim() == 5 && tensor_flags.dim() == 5, "Dimension mismatch");
  AT_ASSERT(tensor_flags.size(1) == 1, "flags is not scalar");
  float bsz = tensor_flags.size(0);
  float d = tensor_flags.size(2);
  float h = tensor_flags.size(3);
  float w = tensor_flags.size(4);

  bool is_3d = (tensor_u.size(1) == 3);
  if (!is_3d) {
     AT_ASSERT(d == 1, "2D velocity field but zdepth > 1");
     AT_ASSERT(tensor_u.size(1) == 2, "2D velocity field must have only 2 channels");
  }
  AT_ASSERT((tensor_u.size(0) == bsz && tensor_u.size(2) == d &&
             tensor_u.size(3) == h && tensor_u.size(4) == w), "Size mismatch");

  AT_ASSERT(tensor_u.is_contiguous() && tensor_flags.is_contiguous(),
            "Input is not contiguous");

  FlagGrid flags(tensor_flags, is_3d);
  MACGrid vel(tensor_u, is_3d);

  at::Backend bckd = flags.getBackend();
  at::ScalarType real = flags.getGridType();
  const T at_zero = getType(bckd, real).scalarTensor(0);

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();
  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;

    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          const bool cur_fluid = flags.isFluid(i, j, k, b);
          const bool cur_obs = flags.isObstacle(i, j, k, b);

          if (!cur_fluid && !cur_obs) {
            continue;
          }

          // we use i > 0 instead of bnd=1 to check outer wall
          if (i > 0 && flags.isObstacle(i - 1, j, k, b)) {
            vel(i, j, k, 0, b) = at_zero;
          }
          if (i > 0 && cur_obs && flags.isFluid(i - 1, j, k, b)) {
            vel(i, j, k, 0, b) = at_zero;
          }
          if (j > 0 && flags.isObstacle(i, j - 1, k, b)) {
            vel(i, j, k, 1, b) = at_zero;
          }
          if (j > 0 && cur_obs && flags.isFluid(i, j - 1, k, b)) {
            vel(i, j, k, 1, b) = at_zero;
          }

          if (k > 0 && flags.isObstacle(i, j, k - 1, b)) {
            vel(i, j, k, 2, b) = at_zero;
          }

          if (k > 0 && cur_obs && flags.isFluid(i, j, k - 1, b)) {
            vel(i, j, k, 2, b) = at_zero;
          }

          if (cur_fluid) {
            if ((i > 0 && flags.isStick(i - 1, j, k, b)) ||
                (i < flags.xsize() - 1 && flags.isStick(i + 1, j, k, b))) {
              vel(i, j, k, 1, b) = at_zero;
              if (vel.is_3d()) {
                vel(i, j, k, 2, b) = at_zero;
              }
            }
            if ((j > 0 && flags.isStick(i, j - 1, k, b)) ||
                (j < flags.ysize() - 1 && flags.isStick(i, j + 1, k, b))) {
              vel(i, j, k, 0, b) = at_zero;
              if (vel.is_3d()) {
                vel(i, j, k, 2, b) = at_zero;
              }
            }
            if (vel.is_3d() &&
                ((k > 0 && flags.isStick(i, j, k - 1, b)) ||
                 (k < flags.zsize() - 1 && flags.isStick(i, j, k + 1, b)))) {
              vel(i, j, k, 0, b) = at_zero;
              vel(i, j, k, 1, b) = at_zero;
            }
          }
        }
      }
    }
  }

}

} // namespace fluid
