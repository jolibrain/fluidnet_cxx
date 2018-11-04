import torch

from . import CellType


def velocityUpdate(pressure, U, flags):

    r""" Calculate the pressure gradient and subtract it into (i.e. calculate
    U' = U - grad(p)). Some care must be taken with handling boundary conditions.
    This function mimics correctVelocity in Manta.
    Velocity update is done IN-PLACE.

    Arguments:
        p (Tensor): scalar pressure field.
        U (Tensor): velocity field (size(2) can be 2 or 3, indicating 2D / 3D)
        flags (Tensor): input occupancy grid
    """
    # Check arguments.
    assert U.dim() == 5 and flags.dim() == 5 and pressure.dim() == 5, \
               "Dimension mismatch"
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
    assert pressure.is_same_size(flags), "size mismatch"
    assert U.is_contiguous() and flags.is_contiguous() and \
              pressure.is_contiguous(), "Input is not contiguous"

    # First, we build the mask for detecting fluid cells. Borders are left untouched.
    # mask_fluid   Fluid cells.
    # mask_fluid_i Fluid cells with (i-1) neighbour also a fluid.
    # mask_fluid_j Fluid cells with (j-1) neighbour also a fluid.
    # mask_fluid_k FLuid cells with (k-1) neighbour also a fluid.

    # Second, we detect obstacle cells
    # See Bridson p44 for algorithm and boundaries treatment.

    if not is3D:
        # Current cell is fluid
        mask_fluid = flags.narrow(4, 1, w-2).narrow(3, 1, h-2).eq(CellType.TypeFluid)
        # Current is fluid and neighbour to left or down are fluid
        mask_fluid_i = mask_fluid.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeFluid))
        mask_fluid_j = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeFluid))
        # Current cell is fluid and neighbours to left or down are obstacle
        mask_fluid_obstacle_im1 = mask_fluid.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeEmpty))
        mask_fluid_obstacle_jm1 = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeEmpty))
        # Current cell is obstacle and not outflow
        mask_obstacle = flags.narrow(4, 1, w-2).narrow(3, 1, h-2) \
                            .eq(CellType.TypeEmpty).__and__ \
                     (flags.narrow(4, 1, w-2).narrow(3, 1, h-2) \
                            .ne(CellType.TypeOutflow))
        # Current cell is obstacle and neighbours to left or down are fluid
        mask_obstacle_fluid_im1 = mask_obstacle.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeFluid))
        mask_obstacle_fluid_jm1 = mask_obstacle.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeFluid))
        # Current cell is obstacle and neighbours to left or down are not fluid
        mask_no_fluid_im1 = mask_obstacle.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).eq(CellType.TypeEmpty))
        mask_no_fluid_jm1 = mask_obstacle.__and__ \
            (flags.narrow(4, 1, w-2).narrow(3, 0, h-2).eq(CellType.TypeEmpty))

    else:
        # TODO: implement 3D bcs well.
        # TODO: add outlfow (change in advection required)
        mask_fluid  = flags.narrow(4, 1, w-2).narrow(3, 1, h-2).narrow(2, 1, d-2).eq(CellType.TypeFluid)
        mask_fluid_i = mask_fluid.__and__ \
            (flags.narrow(4, 0, w-2).narrow(3, 1, h-2).narrow(2, 1, d-2).eq(CellType.TypeFluid))
        mask_fluid_j = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-1).narrow(3, 0, h-1).narrow(2, 1, d-2).eq(CellType.TypeFluid))
        mask_fluid_k = mask_fluid.__and__ \
            (flags.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 0, d-2).eq(CellType.TypeFluid))

    # Cast into float or double tensor and cat into a single mask along chan.
    mask_fluid_i_f = mask_fluid_i.type(U.type())
    mask_fluid_j_f = mask_fluid_j.type(U.type())

    mask_fluid_obstacle_i_f = mask_fluid_obstacle_im1.type(U.type())
    mask_fluid_obstacle_j_f = mask_fluid_obstacle_jm1.type(U.type())

    mask_obstacle_fluid_i_f = mask_obstacle_fluid_im1.type(U.type())
    mask_obstacle_fluid_j_f = mask_obstacle_fluid_jm1.type(U.type())

    mask_no_fluid_i_f = mask_no_fluid_im1.type(U.type())
    mask_no_fluid_j_f = mask_no_fluid_jm1.type(U.type())

    if is3D:
        mask_fluid_k_f = mask_fluid_k.type(U.type())

    if not is3D:
        mask_fluid = torch.cat((mask_fluid_i_f, mask_fluid_j_f), 1).contiguous()
        mask_fluid_obstacle = torch.cat((mask_fluid_obstacle_i_f, mask_fluid_obstacle_j_f), 1).contiguous()
        mask_obstacle_fluid = torch.cat((mask_obstacle_fluid_i_f, mask_obstacle_fluid_j_f), 1).contiguous()
        mask_no_fluid = torch.cat((mask_no_fluid_i_f, mask_no_fluid_j_f), 1).contiguous()
    else:
        mask_fluid = torch.cat((mask_fluid_i_f, mask_fluid_j_f, mask_fluid_k_f), 1).contiguous()

    # pressure tensor.
    #Pijk    Pressure at (i,j,k) in 3 channels (2 for 2D).
    #Pijk_m  Pressure at chan 0: (i-1, j, k)
              #          chan 1: (i, j-1, k)
              #          chan 2: (i, j, k-1)

    if not is3D:
        Pijk = pressure.narrow(4, 1, w-2).narrow(3, 1, h-2)
        Pijk = Pijk.clone().expand(b, 2, d, h-2, w-2)
        Pijk_m = Pijk.clone().expand(b, 2, d, h-2, w-2)
        Pijk_m[:,0] = pressure.narrow(4, 0, w-2).narrow(3, 1, h-2).squeeze(1)
        Pijk_m[:,1] = pressure.narrow(4, 1, w-2).narrow(3, 0, h-2).squeeze(1)
    else:
        Pijk = pressure.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 1, d-2)
        Pijk = Pijk.clone().expand(b, 3, d-2, h-1, w-1)
        Pijk_m = Pijk.clone().expand(b, 3, d-2, h-1, w-1)
        Pijk_m[:,0] = pressure.narrow(4, 0, w-1).narrow(3, 1, h-1).narrow(2, 1, d-2).squeeze(1)
        Pijk_m[:,1] = pressure.narrow(4, 1, w-1).narrow(3, 0, h-1).narrow(2, 1, d-2).squeeze(1)
        Pijk_m[:,2] = pressure.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 0, d-2).squeeze(1)

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

        U[:,:,:,1:(h-1),1:(w-1)] = (mask_fluid * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - (Pijk - Pijk_m)) + \
             mask_fluid_obstacle * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) - Pijk) + \
            mask_obstacle_fluid * \
            (U.narrow(4, 1, w-2).narrow(3, 1, h-2) + Pijk_m) + \
            mask_no_fluid * (0))
        #print('******************************************************')

        #print('masks')
        #print('*    *         *            *')
        #print(mask_fluid)
        #print('*    *         *            *')
        #print(mask_fluid_obstacle)
        #print(mask_obstacle_fluid)
        #print(mask_no_fluid)
        #print('******************************************************')
    else:
        U[:,:,1:(d-1),1:(h-1),1:(w-1)] =  mask * \
            (U.narrow(4, 1, w-1).narrow(3, 1, h-1).narrow(2, 1, d-2) - (Pijk - Pijk_m))

