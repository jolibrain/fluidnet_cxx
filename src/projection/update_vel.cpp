#include "update_vel.h"

namespace fluid {

// *****************************************************************************
// velocityUpdateForward
// *****************************************************************************

void velocityUpdateForward
(
    T& tensor_flags,
    T& tensor_u,
    T& tensor_p,
    const bool is_3d
) {
  // TODO do size checking
  // on the lua stack.

  FlagGrid flags(tensor_flags, is_3d);
  MACGrid vel(tensor_u, is_3d);
  RealGrid pressure(tensor_p, is_3d);

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();
  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;
    const int32_t bnd = 1;
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Manta doesn't touch the velocity on the boundaries (i.e.
            // it stays constant).
            continue;
          }

          if (flags.isFluid(i, j, k, b)) {
            if (flags.isFluid(i - 1, j, k, b)) {
              vel(i, j, k, 0, b) -= (pressure(i, j, k, b) -
                                     pressure(i - 1, j, k, b));
            }
            if (flags.isFluid(i, j - 1, k, b)) {
              vel(i, j, k, 1, b) -= (pressure(i, j, k, b) -
                                     pressure(i, j - 1, k, b));
            }
            if (is_3d && flags.isFluid(i, j, k - 1, b)) {
              vel(i, j, k, 2, b) -= (pressure(i, j, k, b) -
                                     pressure(i, j, k - 1, b));
            }
            if (flags.isEmpty(i - 1, j, k, b)) {
              vel(i, j, k, 0, b) -= pressure(i, j, k, b);
            }
            if (flags.isEmpty(i, j - 1, k, b)) {
              vel(i, j, k, 1, b) -= pressure(i, j, k, b);
            }
            if (is_3d && flags.isEmpty(i, j, k - 1, b)) {
              vel(i, j, k, 2, b) -= pressure(i, j, k, b);
            }
          }
          else if (flags.isEmpty(i, j, k, b) && !flags.isOutflow(i, j, k, b)) {
            // don't change velocities in outflow cells   
            if (flags.isFluid(i - 1, j, k, b)) {
              vel(i, j, k, 0, b) += pressure(i - 1, j, k, b);
            } else {
              vel(i, j, k, 0, b)  = 0;
            }
            if (flags.isFluid(i, j - 1, k, b)) {
              vel(i, j, k, 1, b) += pressure(i, j - 1, k, b);
            } else {
              vel(i, j, k, 1, b)  = 0;
            }
            if (is_3d) {
              if (flags.isFluid(i, j, k - 1, b)) {
                vel(i, j, k, 2, b) += pressure(i, j, k - 1, b);
              } else {
                vel(i, j, k, 2, b)  = 0;
              }
            }
          }
        }
      }
    }
  }
}

} // namespace fluid
