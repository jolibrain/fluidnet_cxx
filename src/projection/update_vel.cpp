#include "update_vel.h"

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
) {
  // Check arguments.
  AT_ASSERT(tensor_u.dim() == 5 && tensor_flags.dim() == 5 && tensor_p.dim() == 5,
             "Dimension mismatch");
  AT_ASSERT(tensor_flags.size(1) == 1, "flags is not scalar");
  int bsz = tensor_flags.size(0);
  int d = tensor_flags.size(2);
  int h = tensor_flags.size(3);
  int w = tensor_flags.size(4);

  bool is_3d = (tensor_u.size(1) == 3);
  if (!is_3d) {
    AT_ASSERT(d == 1, "d > 1 for a 2D domain");
    AT_ASSERT(tensor_u.size(4) == w, "2D velocity field must have only 2 channels");
  }

  AT_ASSERT(tensor_u.size(0) == bsz && tensor_u.size(2) == d && tensor_u.size(3) == h
      && tensor_u.size(4) == w, "size mismatch");
  AT_ASSERT(tensor_p.is_same_size(tensor_flags), "size mismatch");

  AT_ASSERT(tensor_u.is_contiguous() && tensor_flags.is_contiguous() &&
            tensor_p.is_contiguous(), "Input is not contiguous");

  T flags_test = infer_type(tensor_flags).zeros({bsz, 1, d, h, w});
  T flags_test_i = flags_test.clone();
  T flags_test_j = flags_test.clone();
  T flags_test_k = flags_test.clone();

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
            flags_test[b][0][k][j][i] = 1;
            if (flags.isFluid(i - 1, j, k, b)) {
              flags_test_i[b][0][k][j][i] = 1;
              
              vel(i, j, k, 0, b) -= (pressure(i, j, k, b) -
                                     pressure(i - 1, j, k, b));
            }
            if (flags.isFluid(i, j - 1, k, b)) {
             flags_test_j[b][0][k][j][i] = 1;
              vel(i, j, k, 1, b) -= (pressure(i, j, k, b) -
                                     pressure(i, j - 1, k, b));
            }
            if (is_3d && flags.isFluid(i, j, k - 1, b)) {
              flags_test_k[b][0][k][j][i] = 1;
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
