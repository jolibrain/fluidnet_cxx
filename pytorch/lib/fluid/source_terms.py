import torch

from . import CellType
from . import getDx

# *****************************************************************************
# addBuoyancy
# *****************************************************************************

# Add buoyancy force. AddBuoyancy has a dt term.
# Note: Buoyancy is added IN-PLACE.
#
# @input U - vel field (size(2) can be 2 or 3, indicating 2D / 3D)
# @input flags - input occupancy grid
# @input density - scalar density grid.
# @input gravity - 3D vector indicating direction of gravity.
# @input dt - scalar timestep.

def addBuoyancy(U, flags, density, gravity, dt):
    cuda = torch.device('cuda')
    # Argument check
    assert U.dim() == 5 and flags.dim() == 5 and density.dim() == 5,\
        "Dimension mismatch"
    assert flags.size(1) == 1, "flags is not scalar"
    bsz = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    is3D = (U.size(1) == 3)

    bnd = 1

    if not is3D:
        assert d == 1, "2D velocity field but zdepth > 1"
        assert U.size(1) == 2, "2D velocity field must have only 2 channels"

    assert U.size(0) == bsz and U.size(2) == d and \
        U.size(3) == h and U.size(4) == w, "Size mismatch"
    assert density.is_same_size(flags), "Size mismatch"

    assert U.is_contiguous() and flags.is_contiguous() and \
        density.is_contiguous(), "Input is not contiguous"

    assert gravity.dim() == 1 and gravity.size(0) == 3, \
        "Gravity must be a 3D vector (even in 2D)"

    # (aalgua) I don't know why Manta divides by dx, as in all other modules
    # dx = 1.
    strength = gravity * dt

    i = torch.arange(0, w, dtype=torch.long, device=cuda).view(1,w).expand(bsz, d, h, w)
    j = torch.arange(0, h, dtype=torch.long, device=cuda).view(1,h,1).expand(bsz, d, h, w)
    k = torch.zeros_like(i)
    if (is3D):
        k = torch.arange(0, d, dtype=torch.long, device=cuda).view(1,d,1,1).expand(bsz, d, h, w)

    zero = torch.zeros_like(i)
    zeroBy = torch.zeros(i.size(), dtype=torch.uint8, device=cuda)
    zero_f = zero.cuda().float()

    idx_b = torch.arange(start=0, end=bsz, dtype=torch.long, device=cuda) \
                            .view(bsz, 1, 1, 1).expand(bsz,d,h,w)

    maskBorder = (i < bnd).__or__\
        (i > w - 1 - bnd).__or__\
        (j < bnd).__or__\
        (j > h - 1 - bnd)
    if (is3D):
        maskBorder = maskBorder.__or__(k < bnd).__or__\
            (k > d - 1 - bnd)

    maskBorder = maskBorder.unsqueeze(1)

    # No buoyancy on the border. Set continue (mCont) to false.
    mCont = torch.ones_like(zeroBy).unsqueeze(1)
    mCont.masked_fill_(maskBorder, 0)

    isFluid = flags.eq(CellType.TypeFluid).__and__(mCont)
    mCont.masked_fill_(isFluid.ne(1), 0)
    mCont.squeeze_(1)

    max_X = torch.zeros_like(zero).fill_(w-1)
    max_Y = torch.zeros_like(zero).fill_(h-1)

    i_l = zero.where( (i <= 0), i-1)
    i_r = max_X.where( (i > w-2), i+1)
    j_l = zero.where( (j <= 0), j-1)
    j_r = max_Y.where( (j > h-2), j+1)

    rho_star = 1.0

    fluid100 =  flags[idx_b, zero, k, j, i_l].eq(CellType.TypeFluid).__and__ \
                (mCont)
               #(flags[idx_b, zero, k, j, i_r].eq(CellType.TypeFluid)).__and__ \

    factor = strength[0] * ((0.5* \
        (density[idx_b, zero, k, j, i] +
        density[idx_b, zero, k, j, i_l] ) - rho_star))
    U[:,0].masked_scatter_(fluid100, (U.select(1,0) + factor).masked_select(fluid100))

    fluid010 =  flags[idx_b, zero, k, j_l, i].eq(CellType.TypeFluid).__and__ \
                (mCont)
               #(flags[idx_b, zero, k, j_r, i].eq(CellType.TypeFluid)).__and__ \
    #fluid010 = zeroBy.where( j <= 0, (flags[idx_b, zero, k, j-1, i].eq(CellType.TypeFluid))).__and__(mCont)
    factor = strength[1] * ((0.5* \
        (density[idx_b, zero, k, j, i] +
        density[idx_b, zero, k, j_l, i] ) - rho_star))
    #factor = strength[1] * (density.squeeze(1) - \
    #    zero_f.where( j <= 0, density[idx_b, zero, k, j-1, i]) )
    U[:,1].masked_scatter_(fluid010, (U.select(1,1) + factor).masked_select(fluid010))

    if (is3D):
        fluid001 = zeroBy.where( j <= 0, (flags[idx_b, zero, k-1, j, i].eq(CellType.TypeFluid))).__and__(mCont)
        factor = strength[2] *(0.5* (density.squeeze(1) + \
            zero_f.where(k <= 1, density[idx_b, zero, k-1, j, i]) ))
        U[:,2].masked_scatter_(fluid001, (U.select(1,2) + factor).masked_select(fluid001))

    return U

# *****************************************************************************
# addGravity
# *****************************************************************************

# Add gravity force. It has a dt term.
# Note: gravity is added IN-PLACE.
#
# @input U - vel field (size(2) can be 2 or 3, indicating 2D / 3D)
# @input flags - input occupancy grid
# @input gravity - 3D vector indicating direction of gravity.
# @input dt - scalar timestep.

def addGravity(U, flags, gravity, dt):

    cuda = torch.device('cuda')
    # Argument check
    assert U.dim() == 5 and flags.dim() == 5, "Dimension mismatch"
    assert flags.size(1) == 1, "flags is not scalar"
    bsz = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    is3D = (U.size(1) == 3)

    bnd = 1
    if not is3D:
        assert d == 1, "2D velocity field but zdepth > 1"
        assert U.size(1) == 2, "2D velocity field must have only 2 channels"

    assert U.size(0) == bsz and U.size(2) == d and \
        U.size(3) == h and U.size(4) == w, "Size mismatch"

    assert U.is_contiguous() and flags.is_contiguous(), "Input is not contiguous"

    assert gravity.dim() == 1 and gravity.size(0) == 3,\
        "Gravity must be a 3D vector (even in 2D)"

    # (aalgua) I don't know why Manta divides by dx, as in all other modules
    # dx = 1.
    force = gravity * dt

    i = torch.arange(0, w, dtype=torch.long, device=cuda).view(1,w).expand(bsz, d, h, w)
    j = torch.arange(0, h, dtype=torch.long, device=cuda).view(1,h,1).expand(bsz, d, h, w)
    k = torch.zeros_like(i)
    if (is3D):
        k = torch.arange(0, d, dtype=torch.long, device=cuda).view(1,d,1,1).expand(bsz, d, h, w)

    zero = torch.zeros_like(i)
    zeroBy = torch.zeros(i.size(), dtype=torch.uint8, device=cuda)
    zero_f = zero.float()

    idx_b = torch.arange(start=0, end=bsz, dtype=torch.long, device=cuda) \
                            .view(bsz, 1, 1, 1).expand(bsz,d,h,w)

    maskBorder = (i < bnd).__or__\
        (i > w - 1 - bnd).__or__\
        (j < bnd).__or__\
        (j > h - 1 - bnd)
    if (is3D):
        maskBorder = maskBorder.__or__(k < bnd).__or__(k > d - 1 - bnd)

    maskBorder = maskBorder.unsqueeze(1)

    # No buoyancy on the border. Set continue (mCont) to false.
    mCont = torch.ones_like(zeroBy).unsqueeze(1)
    mCont.masked_fill_(maskBorder, 0)

    cur_fluid = flags.eq(CellType.TypeFluid).__and__(mCont)
    cur_empty = flags.eq(CellType.TypeEmpty).__and__(mCont)

    mNotFluidNotEmpt = cur_fluid.ne(1).__and__(cur_empty.ne(1))
    mCont.masked_fill_(mNotFluidNotEmpt, 0)

    mCont.squeeze_(1)
    #print()
    #print('F = ')
    #print(force)

    #print('before')
    #print(U)
    #print(U.size())
    fluid100 = (zeroBy.where( i <= 0, (flags[idx_b, zero, k, j, i-1].eq(CellType.TypeFluid))) \
    .__or__(( zeroBy.where( i <= 0, (flags[idx_b, zero, k, j, i-1].eq(CellType.TypeEmpty)))) \
    .__and__(cur_fluid.squeeze(1)))).__and__(mCont)
    U[:,0].masked_scatter_(fluid100, (U[:,0] + force[0]).masked_select(fluid100))

    fluid010 = (zeroBy.where( j <= 0, (flags[idx_b, zero, k, j-1, i].eq(CellType.TypeFluid))) \
    .__or__(( zeroBy.where( j <= 0, (flags[idx_b, zero, k, j-1, i].eq(CellType.TypeEmpty)))) \
    .__and__(cur_fluid.squeeze(1))) ).__and__(mCont)
    U[:,1].masked_scatter_(fluid010, (U[:,1] + force[1]).masked_select(fluid010))

    if (is3D):
        fluid001 = (zeroBy.where( k <= 0, (flags[idx_b, zero, k-1, j, i].eq(CellType.TypeFluid))) \
        .__or__(( zeroBy.where( k <= 0, (flags[idx_b, zero, k-1, j, i].eq(CellType.TypeEmpty)))) \
        .__and__(cur_fluid.squeeze(1)))).__and__(mCont)
        U[:,2].masked_scatter_(fluid001, (U[:,2] + force[2]).masked_select(fluid001))
    #print('after')
    #print(U)
    #print(U.size())
    return U
