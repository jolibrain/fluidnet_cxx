#include "div.h"

namespace fluid {

// *****************************************************************************
// velocityDivergenceForward
// *****************************************************************************

void velocityDivergenceForward
(
    T& tensor_u,
    T& tensor_flags,
    T& tensor_u_div
) {
  // Check sizes
  AT_ASSERT(tensor_u.dim() == 5 && tensor_flags.dim() == 5 && tensor_u_div.dim() == 5,
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
  AT_ASSERT(tensor_flags.dim() == tensor_u_div.dim(), "Size mismatch");

  AT_ASSERT(tensor_u.is_contiguous() && tensor_flags.is_contiguous() &&
            tensor_u_div.is_contiguous(), "Input is not contiguous");

  FlagGrid flags(tensor_flags, is_3d);
  MACGrid  vel(tensor_u, is_3d);
  RealGrid rhs(tensor_u_div, is_3d);

  const int32_t xsize = flags.xsize();
  const int32_t ysize = flags.ysize();
  const int32_t zsize = flags.zsize();
  const int32_t nbatch = flags.nbatch();

  for (int32_t b = 0; b < nbatch; b++) {
    int32_t k, j, i;
    const int32_t bnd = 1;

    // Note: our kernel assumes enforceCompatibility == false (i.e. we do not
    // do the reduction) and that fractions are not provided.
    for (k = 0; k < zsize; k++) {
      for (j = 0; j < ysize; j++) {
        for (i = 0; i < xsize; i++) {
          if (i < bnd || i > xsize - 1 - bnd ||
              j < bnd || j > ysize - 1 - bnd ||
              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
            // Manta zeros stuff on the border.
            rhs(i, j, k, b) = 0;
            continue;
          }

          if (!flags.isFluid(i, j, k, b)) {
            rhs(i, j, k, b) = 0;
            continue;
          }

          // compute divergence (rhs of poisson equation) 
          // no flag checks: assumes vel at obstacle interfaces is set to zero.
          T div =
              vel(i, j, k, 0, b) - vel(i + 1, j, k, 0, b) +
              vel(i, j, k, 1, b) - vel(i, j + 1, k, 1, b);
          if (is_3d) {
            div += (vel(i, j, k, 2, b) - vel(i, j, k + 1, 2, b));
          }
          rhs(i, j, k, b) = div;
        }
      }
    }
  }
}

} // namespace fluid  
