#include "source_term.h"

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
void addBuoyancy
(
   T& tensor_u,
   T& tensor_flags,
   T& tensor_density,
   T& tensor_gravity,
   const float dt
) {
  // Argument check
  AT_ASSERT(tensor_u.dim() == 5 && tensor_flags.dim() == 5 && tensor_density.dim() == 5,
    "Dimension mismatch");
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
  AT_ASSERT(tensor_density.is_same_size(tensor_flags), "Size mismatch");

  AT_ASSERT(tensor_u.is_contiguous() && tensor_flags.is_contiguous() &&
            tensor_density.is_contiguous(), "Input is not contiguous");

  AT_ASSERT(tensor_gravity.dim() == 1 && tensor_gravity.size(0) == 3,
           "Gravity must be a 3D vector (even in 2D)");
  
  FlagGrid flags(tensor_flags, is_3d);
  MACGrid vel(tensor_u, is_3d);
  RealGrid factor(tensor_density, is_3d);

  T strength = -tensor_gravity * (dt / flags.getDx());

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();

  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;
    const int32_t bnd = 1;
    // Note: our kernel assumes enforceCompatibility == false (i.e. we do not
    // do the reduction) and that fractions are not provided.
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // No buoyancy on the border.
            continue;
          }

          if (!flags.isFluid(i, j, k, b)) {
            continue;
          }
          if (flags.isFluid(i - 1, j, k, b)) {
            vel(i, j, k, 0, b) += (0.5 * strength[0] *
                                (factor(i, j, k, b) + factor(i - 1, j, k, b)));
          }
          if (flags.isFluid(i, j - 1, k, b)) {
            vel(i, j, k, 1, b) += (0.5 * strength[1] *
                                (factor(i, j, k, b) + factor(i, j - 1, k, b)));
          }
          if (is_3d && flags.isFluid(i, j, k - 1, b)) {
            vel(i, j, k, 2, b) += (0.5 * strength[2] *
                                (factor(i, j, k, b) + factor(i, j, k - 1, b)));
          }
        }
      }
    }
  }
}

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

void addGravity
(
   T& tensor_u,
   T& tensor_flags,
   T& tensor_gravity,
   const float dt
) {
  // Argument check
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

  AT_ASSERT(tensor_gravity.dim() == 1 && tensor_gravity.size(0) == 3,
           "Gravity must be a 3D vector (even in 2D)");

  FlagGrid flags(tensor_flags, is_3d);
  MACGrid vel(tensor_u, is_3d);

  T force = tensor_gravity * (dt / flags.getDx());

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();

  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;
    const int32_t bnd = 1;
#pragma omp parallel for collapse(3) private(k, j, i)
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // No buoyancy on the border.
            continue;
          }

          const bool curFluid = flags.isFluid(i, j, k, b);
          const bool curEmpty = flags.isEmpty(i, j, k, b);

          if (!curFluid && !curEmpty) {
            continue;
          }

          if (flags.isFluid(i - 1, j, k, b) ||
              (curFluid && flags.isEmpty(i - 1, j, k, b))) {
            vel(i, j, k, 0, b) += force[0];
          }

          if (flags.isFluid(i, j - 1, k, b) ||
              (curFluid && flags.isEmpty(i, j - 1, k, b))) {
            vel(i, j, k, 1, b) += force[1];
          }

          if (flags.is_3d() && (flags.isFluid(i, j, k - 1, b) ||
              (curFluid && flags.isEmpty(i, j, k - 1, b)))) {
            vel(i, j, k, 2, b) += force[2];
          }
        }
      }
    }
  }
}

} // namespace fluid
