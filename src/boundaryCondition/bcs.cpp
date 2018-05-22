#include "bcs.h"

namespace fluid {

// *****************************************************************************
// setWallBcsForward
// *****************************************************************************

void setWallBcsForward
(
    T& tensor_flags,
    T& tensor_u,
    const bool is_3d
) {
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
