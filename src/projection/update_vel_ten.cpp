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
    T& U,
    T& flags,
    T& pressure
) {
  // Check arguments.
  AT_ASSERT(U.dim() == 5 && flags.dim() == 5 && pressure.dim() == 5,
             "Dimension mismatch");
  AT_ASSERT(flags.size(1) == 1, "flags is not scalar");
  int b = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);

  bool is3D = (U.size(1) == 3);
  if (!is3D) {
    AT_ASSERT(d == 1, "d > 1 for a 2D domain");
    AT_ASSERT(U.size(4) == w, "2D velocity field must have only 2 channels");
  }

  AT_ASSERT(U.size(0) == b && U.size(2) == d && U.size(3) == h
      && U.size(4) == w, "size mismatch");
  AT_ASSERT(pressure.is_same_size(flags), "size mismatch");

  AT_ASSERT(U.is_contiguous() && flags.is_contiguous() &&
            pressure.is_contiguous(), "Input is not contiguous");

  // First, we build the mask for detecting fluid cells. Borders are left untouched.
  at::Tensor mask_fluid;   // Fluid cells.
  at::Tensor mask_fluid_i; // Fluid cells with (i-1) neighbour also a fluid. 
  at::Tensor mask_fluid_j; // Fluid cells with (j-1) neighbour also a fluid.
  at::Tensor mask_fluid_k; // FLuid cells with (k-1) neighbour also a fluid.

  if (!is3D) {
    mask_fluid = flags.narrow(4, 1, w-2).narrow(3, 1, h-2).eq(fluid::TypeFluid);
    mask_fluid_i = mask_fluid.__and__
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(fluid::TypeFluid));
    mask_fluid_j = mask_fluid.__and__
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(fluid::TypeFluid));
  } else {
    mask_fluid  = flags.narrow(4, 1, w-2).narrow(3, 1, h-2).narrow(2, 1, d-2).eq(fluid::TypeFluid);
    mask_fluid_i = mask_fluid.__and__
     (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).narrow(2, 1, d-2).eq(fluid::TypeFluid));
    mask_fluid_j = mask_fluid.__and__
     (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).narrow(2, 1, d-2).eq(fluid::TypeFluid));
    mask_fluid_k = mask_fluid.__and__
     (flags.narrow(4, 1, w-2).narrow(3, 1, h-2).narrow(2, 0, d-2).eq(fluid::TypeFluid));
  }

  // Cast into float or double tensor and cat into a single mask along chan.
  at::Tensor mask_fluid_i_f = mask_fluid_i.type().toScalarType(U.type().scalarType())
                                .copy(mask_fluid_i);
  at::Tensor mask_fluid_j_f = mask_fluid_j.type().toScalarType(U.type().scalarType())
                                .copy(mask_fluid_j);
  at::Tensor mask_fluid_k_f;
  if (is3D) {
    mask_fluid_k_f = mask_fluid_k.type().toScalarType(U.type().scalarType())
                       .copy(mask_fluid_k);
  }

  at::Tensor mask;
  if(!is3D) {
     mask = at::cat({mask_fluid_i_f, mask_fluid_j_f}, 1).contiguous();
  } else {
     mask = at::cat({mask_fluid_i_f, mask_fluid_j_f, mask_fluid_k_f}, 1).contiguous();
  }

  // pressure tensor.
  at::Tensor Pijk; // Pressure at (i,j,k) in 3 channels (2 for 2D).
  at::Tensor Pijk_m; // Pressure at chan 0: (i-1, j, k)
                     //             chan 1: (i, j-1, k)
                     //             chan 2: (i, j, k-1)

  if (!is3D) {
    Pijk = pressure.narrow(4, 1, w-2).narrow(3, 1, h-2);
    Pijk = Pijk.clone().expand({b, 2, d, h-2, w-2});
    Pijk_m = Pijk.clone().expand({b, 2, d, h-2, w-2});
    Pijk_m.select(1,0) = pressure.narrow(4, 0, w-2).narrow(3, 1, h-2).squeeze(1);
    Pijk_m.select(1,1) = pressure.narrow(4, 1, w-2).narrow(3, 0, h-2).squeeze(1);
  } else {
    Pijk = pressure.narrow(4, 1, w-2).narrow(3, 1, h-2).narrow(2, 1, d-2);
    Pijk = Pijk.clone().expand({b, 3, d-2, h-2, w-2});
    Pijk_m = Pijk.clone().expand({b, 3, d-2, h-2, w-2});
    Pijk_m.select(1,0) = pressure.narrow(4, 0, w-2).narrow(3, 1, h-2).narrow(2, 1, d-2).squeeze(1);
    Pijk_m.select(1,1) = pressure.narrow(4, 1, w-2).narrow(3, 0, h-2).narrow(2, 1, d-2).squeeze(1);
    Pijk_m.select(1,2) = pressure.narrow(4, 1, w-2).narrow(3, 1, h-2).narrow(2, 0, d-2).squeeze(1);
  }

  // u = u - grad(p)
  // grad(p) = [[ p(i,j,k) - p(i-1,j,k) ]
  //            [ p(i,j,k) - p(i,j-1,k) ]
  //            [ p(i,j,k) - p(i,j,k-1) ]]
  if (!is3D) {
    U.narrow(4, 1, w-2).narrow(3, 1, h-2) = mask *
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m));
  } else {
    U.narrow(4, 1, w-2).narrow(3, 1, h-2).narrow(2, 1, d-2) =  mask *
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2).narrow(2, 1, d-2) - (Pijk - Pijk_m));
  }

}

} // namespace fluid
