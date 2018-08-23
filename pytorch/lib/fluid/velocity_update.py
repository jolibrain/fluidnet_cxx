import torch

from . import CellType

# *****************************************************************************
# velocityUpdate
# *****************************************************************************

# Calculate the pressure gradient and subtract it into (i.e. calculate
# U' = U - grad(p)). Some care must be taken with handling boundary conditions.
# This function mimics correctVelocity in Manta.
# Velocity update is done IN-PLACE.
#
# input p - scalar pressure field.
# input U - vel field (size(2) can be 2 or 3, indicating 2D / 3D)
# input flags - input occupancy grid

def velocityUpdate(dt, pressure, U, flags):
    # Check arguments.
    assert U.dim() == 5 and flags.dim() == 5 and pressure.dim() == 5, \
               "Dimension mismatch"
    assert flags.size(1) == 1, "flags is not scalar"
    b = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    dt = float(dt)

    is3D = (U.size(1) == 3)
    if not is3D:
        assert d == 1, "d > 1 for a 2D domain"
        assert U.size(4) == w, "2D velocity field must have only 2 channels"

    assert U.size(0) == b and U.size(2) == d and U.size(3) == h \
        and U.size(4) == w, "size mismatch"
    assert pressure.is_same_size(flags), "size mismatch"
    assert U.is_contiguous() and flags.is_contiguous() and \
              pressure.is_contiguous(), "Input is not contiguous"

    # First, we build the mask for detecting fluid cells. Borders are left untouched.
    # mask_fluid   Fluid cells.
    # mask_fluid_i Fluid cells with (i-1) neighbour also a fluid.
    # mask_fluid_j Fluid cells with (j-1) neighbour also a fluid.
    # mask_fluid_k FLuid cells with (k-1) neighbour also a fluid.

    # Second, we detect obstacle cells (in Manta they do it with empty cells, but it
    # is clear that there must be a treatment of solid cells which are not empty.
    # See Bridson p44 for algorithm and boundaries treatment.

    if not is3D:
        # Current cell is fluid
        mask_fluid = flags.narrow(4, 1, w-1).narrow(3, 1, h-1).eq(CellType.TypeFluid)
        # Current is fluid and neighbour to left or down are fluid
        mask_fluid_i = mask_fluid.__and__ \
            (flags.narrow(4, 0, w-1).narrow(3, 1, h-1).eq(CellType.TypeFluid))
        mask_fluid_j = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-1).narrow(3, 0, h-1).eq(CellType.TypeFluid))
        # Current cell is fluid and neighbours to left or down are obstacle
        mask_fluid_obs_im1 = mask_fluid.__and__ \
            (flags.narrow(4, 0, w-1).narrow(3, 1, h-1).eq(CellType.TypeObstacle))
        mask_fluid_obs_jm1 = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-1).narrow(3, 0, h-1).eq(CellType.TypeObstacle))
        # Current cell is obstacle
        mask_obs = flags.narrow(4, 1, w-1).narrow(3, 1, h-1).eq(CellType.TypeObstacle)
        # Current cell is obstacle and neighbours to left or down are fluid
        mask_obs_fluid_im1 = mask_obs.__and__ \
            (flags.narrow(4, 0, w-1).narrow(3, 1, h-1).eq(CellType.TypeFluid))
        mask_obs_fluid_jm1 = mask_obs.__and__ \
            (flags.narrow(4, 1, w-1).narrow(3, 0, h-1).eq(CellType.TypeFluid))

    else:
        # TODO: implement 3D bcs well.
        # TODO: add outlfow (change in advection required)
        mask_fluid  = flags.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 1, d-1).eq(CellType.TypeFluid)
        mask_fluid_i = mask_fluid.__and__ \
            (flags.narrow(4, 0, w-1).narrow(3, 1, h-1).narrow(2, 1, d-1).eq(CellType.TypeFluid))
        mask_fluid_j = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-1).narrow(3, 0, h-1).narrow(2, 1, d-1).eq(CellType.TypeFluid))
        mask_fluid_k = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 0, d-1).eq(CellType.TypeFluid))

    # Cast into float or double tensor and cat into a single mask along chan.
    mask_fluid_i_f = mask_fluid_i.type(U.type())
    mask_fluid_j_f = mask_fluid_j.type(U.type())

    mask_fluid_obs_i_f = mask_fluid_obs_im1.type(U.type())
    mask_fluid_obs_j_f = mask_fluid_obs_jm1.type(U.type())

    mask_obs_fluid_i_f = mask_obs_fluid_im1.type(U.type())
    mask_obs_fluid_j_f = mask_obs_fluid_jm1.type(U.type())

    if is3D:
        mask_fluid_k_f = mask_fluid_k.type(U.type())

    if not is3D:
        mask_fluid = torch.cat((mask_fluid_i_f, mask_fluid_j_f), 1).contiguous()
        mask_fluid_obs = torch.cat((mask_fluid_obs_i_f, mask_fluid_obs_j_f), 1).contiguous()
        mask_obs_fluid = torch.cat((mask_obs_fluid_i_f, mask_obs_fluid_j_f), 1).contiguous()
    else:
        mask_fluid = torch.cat((mask_fluid_i_f, mask_fluid_j_f, mask_fluid_k_f), 1).contiguous()

    # pressure tensor.
    #Pijk    Pressure at (i,j,k) in 3 channels (2 for 2D).
    #Pijk_m  Pressure at chan 0: (i-1, j, k)
              #          chan 1: (i, j-1, k)
              #          chan 2: (i, j, k-1)

    if not is3D:
        Pijk = pressure.narrow(4, 1, w-1).narrow(3, 1, h-1)
        Pijk = Pijk.clone().expand(b, 2, d, h-1, w-1)
        Pijk_m = Pijk.clone().expand(b, 2, d, h-1, w-1)
        Pijk_m[:,0] = pressure.narrow(4, 0, w-1).narrow(3, 1, h-1).squeeze(1)
        Pijk_m[:,1] = pressure.narrow(4, 1, w-1).narrow(3, 0, h-1).squeeze(1)
    else:
        Pijk = pressure.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 1, d-1)
        Pijk = Pijk.clone().expand(b, 3, d-1, h-1, w-1)
        Pijk_m = Pijk.clone().expand(b, 3, d-1, h-1, w-1)
        Pijk_m[:,0] = pressure.narrow(4, 0, w-1).narrow(3, 1, h-1).narrow(2, 1, d-1).squeeze(1)
        Pijk_m[:,1] = pressure.narrow(4, 1, w-1).narrow(3, 0, h-1).narrow(2, 1, d-1).squeeze(1)
        Pijk_m[:,2] = pressure.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 0, d-1).squeeze(1)

    # grad(p) = [[ p(i,j,k) - p(i-1,j,k) ]
    #            [ p(i,j,k) - p(i,j-1,k) ]
    #            [ p(i,j,k) - p(i,j,k-1) ]]
    if not is3D:
        # Three cases:
        # 1) Cell is fluid and left neighbour is fluid:
        # u = u - grad(p)
        # 2) Cell is fluid and left neighbour is obstacle
        # u = u - p(i,j)
        # 3) Cell is obstacle and left neighbour is fluid
        # u = u + p(i-1,j)
        U[:,:,:,1:h,1:w] = dt * (mask_fluid * \
            (U.narrow(4, 1, w-1).narrow(3, 1, h-1) - (Pijk - Pijk_m)) + \
             mask_fluid_obs * \
            (U.narrow(4, 1, w-1).narrow(3, 1, h-1) - Pijk) + \
            mask_obs_fluid * \
            (U.narrow(4, 1, w-1).narrow(3, 1, h-1) + Pijk_m))
    else:
        U[:,:,1:(d-1),1:(h-1),1:(w-1)] =  mask * \
            (U.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 1, d-1) - (Pijk - Pijk_m))

