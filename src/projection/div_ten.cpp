#include "div.h"

namespace fluid {

// *****************************************************************************
// velocityDivergenceForward
// *****************************************************************************

// Calculate the velocity divergence (with boundary cond modifications). This is
// essentially a replica of makeRhs in Manta and FluidNet.
// 
// input U - input vel field (size(2) can be 2 or 3, indicating 2D / 3D)
// input flags - input occupancy grid
// input UDiv - output divergence (scalar field). 

void velocityDivergenceForward(T& U, T& flags, T& UDiv) {
  // Check sizes
  AT_ASSERT(U.dim() == 5 && flags.dim() == 5 && UDiv.dim() == 5,
    "Dimension mismatch");
  AT_ASSERT(flags.size(1) == 1, "flags is not scalar");
  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);  
  int w = flags.size(4);
  int bnd = 1; // Boundary width (hard coded)

  int z = 2;
  int y = 3;
  int x = 4;
  
  bool is_3d = (U.size(1) == 3);
  if (!is_3d) {
     AT_ASSERT(d == 1, "2D velocity field but zdepth > 1");
     AT_ASSERT(U.size(1) == 2, "2D velocity field must have only 2 channels"); 
  }
  AT_ASSERT((U.size(0) == bsz && U.size(z) == d &&
             U.size(y) == h && U.size(x) == w), "Size mismatch");
  AT_ASSERT(UDiv.is_same_size(flags), "Size mismatch");

  AT_ASSERT(U.is_contiguous() && flags.is_contiguous() &&
            UDiv.is_contiguous(), "Input is not contiguous");
  
  T Uijk; // Velocity in ijk
  T Uijk_p; // Velocity in (i+1),(j+1),(k+1)

  // Remove the borders in x, y and z and build the i+1, j+1, k+1 tensor
  if (!is_3d) {
    Uijk = U.narrow(x, 1, w-2).narrow(y, 1, h-2);
    Uijk_p = Uijk.clone();
    Uijk_p.select(1,0) = U.narrow(x, 2, w-2).narrow(y, 1, h-2).select(1,0);
    Uijk_p.select(1,1) = U.narrow(x, 1, w-2).narrow(y, 2, h-2).select(1,1);
  } else {
    Uijk = U.narrow(x, 1, w-2).narrow(y, 1, h-2).narrow(z, 1, d-2);
    Uijk_p = Uijk.clone();
    Uijk_p.select(1,0) = U.narrow(x, 2, w-2).narrow(y, 1, h-2).narrow(z, 1, d-2).select(1,0);
    Uijk_p.select(1,1) = U.narrow(x, 1, w-2).narrow(y, 2, h-2).narrow(z, 1, d-2).select(1,1);
    Uijk_p.select(1,2) = U.narrow(x, 1, w-2).narrow(y, 1, h-2).narrow(z, 2, d-2).select(1,2);
  }

  // -div = u(i+1,j,k) - u(i,j,k) +
  //        v(i,j+1,k) - v(i,j,k) +
  //        w(i,j,k+1) - w(i,j,k) 
                       
  at::Tensor div = Uijk.select(1,0) - Uijk_p.select(1,0) +
                   Uijk.select(1,1) - Uijk_p.select(1,1);

  if (is_3d) {
    div += Uijk.select(1,2) - Uijk_p.select(1,2);
  }

  if (!is_3d) {
    UDiv.narrow(x, 1, w-2).narrow(y, 1, h-2) = div.view({bsz, 1, d, h-2, w-2});
  } else {
    UDiv.narrow(x, 1, w-2).narrow(y, 1, h-2).narrow(z, 1, d-2) = div.view({bsz, 1, d-2, h-2, w-2});
  }

  //Set div to 0 in obstacles
  at::Tensor mask_obst = flags.eq(TypeObstacle);
  UDiv.masked_fill_(mask_obst, 0);

}

} // namespace fluid  
