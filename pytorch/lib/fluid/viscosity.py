import torch

from . import CellType

# *****************************************************************************
# addViscosity
# *****************************************************************************

# ******WARNING********: ONLY IN 2D
# Adds viscosity to velocity field.
# Velocity update is done IN-PLACE.
#
# input dt - time step (in seconds)
# input U - vel field (size(2) can be 2 or 3, indicating 2D / 3D)
# input flags - input occupancy grid
# input viscosity - viscosity

def addViscosity(dt, U, flags, viscosity):
    # Check arguments.
    assert U.dim() == 5 and flags.dim() == 5, "Dimension mismatch"
    assert flags.size(1) == 1, "flags is not scalar"
    b = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    is3D = (U.size(1) == 3)
    if not is3D:
        assert d == 1, "d > 1 for a 2D domain"
        assert U.size(4) == w, "2D velocity field must have only 2 channels"

    assert U.size(0) == b and U.size(2) == d and U.size(3) == h \
        and U.size(4) == w, "size mismatch"
    assert U.is_contiguous() and flags.is_contiguous(), "Input is not contiguous"

    # First, we build the mask for detecting fluid cells. Borders are left untouched.
    # mask_fluid   Fluid cells.
    # mask_fluid_i Fluid cells with (i-1) neighbour also a fluid.
    # mask_fluid_j Fluid cells with (j-1) neighbour also a fluid.
    # mask_fluid_k Fluid cells with (k-1) neighbour also a fluid.

    if not is3D:
        mask_fluid = flags.narrow(4, 1, w-2).narrow(3, 1, h-2).eq(CellType.TypeFluid)
        mask_fluid_i = mask_fluid.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeFluid))
        mask_fluid_j = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeFluid))

    # Cast into float or double tensor and cat into a single mask along chan.
    mask_fluid_i_f = mask_fluid_i.type(U.type())
    mask_fluid_j_f = mask_fluid_j.type(U.type())
    if is3D:
        mask_fluid_k_f = mask_fluid_k.type(U.type())

    if not is3D:
        mask = torch.cat((mask_fluid_i_f, mask_fluid_j_f), 1).contiguous()
    else:
        mask = torch.cat((mask_fluid_i_f, mask_fluid_j_f, mask_fluid_k_f), 1).contiguous()

    # Update the velocity, to the viscous velocity field.
    # u^v(i,j) = u^n(i,j)
    #      + dt*nu*[u^n(i+1,j) + u^n(i,j+1) + u^n(i-1,j) + u^n(i,j-1)
    #                   -4u^n(i,j) ]
    if not is3D:
        U[:,:,:,1:(h-1),1:(w-1)] = mask * (\
            U.narrow(4, 1, w-2).narrow(3, 1, h-2) +
            dt * viscosity *(
            U.narrow(4, 2, w-2).narrow(3, 1, h-2) + \
            U.narrow(4, 1, w-2).narrow(3, 2, h-2) + \
            U.narrow(4, 0, w-2).narrow(3, 1, h-2) + \
            U.narrow(4, 0, w-2).narrow(3, 0, h-2) - \
            (4*U.narrow(4, 1, w-2).narrow(3, 1, h-2) )\
            ) )
