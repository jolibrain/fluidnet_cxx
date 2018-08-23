import torch

from . import CellType

def setWallBcsStick(U, flags, flags_stick):

    cuda = torch.device('cuda')
    assert (U.dim() == 5 and flags.dim() == 5 and flags_stick.dim() == 5), 'Dimension mismatch'
    assert flags.size(1) == 1, 'flags is not a scalar'
    assert flags_stick.size(1) == 1, 'flags is not a scalar'
    bsz = flags.size(0)
    d = flags.size(2)
    h = flags.size(3)
    w = flags.size(4)

    is3D = (U.size(1) == 3)
    if (not is3D):
        assert d == 1, '2D velocity field but zdepth > 1'
        assert U.size(1) == 2, '2D velocity field must have only 2 channels'

    assert (U.size(0) == bsz and U.size(2) == d and U.size(3) == h and U.size(4) == w),\
        'Size mismatch'
    assert (U.is_contiguous() and flags.is_contiguous() and flags_stick.is_contiguous()), 'Input is not contiguous'

    i = torch.arange(start=0, end=w, dtype=torch.long, device=cuda) \
            .view(1,w).expand(bsz, d, h, w)
    j = torch.arange(start=0, end=h, dtype=torch.long, device=cuda) \
            .view(1,h,1).expand(bsz, d, h, w)
    k = torch.zeros_like(i)
    if (is3D):
        k = torch.arange(start=0, end=d, dtype=torch.long, device=cuda) \
                .view(1,d,1,1).expand(bsz, d, h, w)

    zero = torch.zeros_like(i)
    maxX = torch.zeros_like(i).fill_(w-1)
    maxY = torch.zeros_like(i).fill_(h-1)
    zeroF = zero.float()
    zeroBy = torch.zeros(i.size(), dtype=torch.uint8, device=cuda)

    idx_b = torch.arange(start=0, end=bsz, dtype=torch.long, device=cuda) \
                .view(bsz, 1, 1, 1).expand(bsz,d,h,w)

    mCont = torch.ones_like(zeroBy)

    cur_fluid = flags.eq(CellType.TypeFluid).squeeze(1)
    cur_obs = flags.eq(CellType.TypeObstacle).squeeze(1)
    cur_stick = flags_stick.eq(CellType.TypeStick).squeeze(1)

    stick_is_obs = cur_obs.masked_select(cur_stick.eq(CellType.TypeStick).squeeze(1))
    assert stick_is_obs.all(), 'Stick cells must be also obstacles.'
    mNotCells = cur_fluid.ne(1).__and__\
                (cur_obs.ne(1)).__and__\
                (cur_stick.ne(1))
    mCont.masked_fill_(mNotCells, 0)

    # First, set all velocities INSIDE obstacles to zero.
    U.masked_fill_(cur_obs.unsqueeze(1), 0)

    # Slip Boundary Conditon. Normal vel is 0.
    # The neighbour to the left (i-1,j,k) is an obstacle
    im1 = zero.where(i <=0, i - 1)
    obst_im1jk = zeroBy.where(i <= 0, (flags[idx_b, zero, k, j, im1].eq(TypeObstacle))).__and__(mCont)
    U[:,0].masked_fill_(obst_im1jk, 0)

    # Current cell is an obstacle.
    # The neighbour to the left (i-1,j,k) is fluid.
    # Set normal direction velocity to 0.
    obs_ijk_fluid_im1jk = zeroBy.where(i <= 0, (flags[idx_b, zero, k, j, im1].eq(TypeFluid))). \
     __and__(cur_obs).__and__(mCont)
    U[:,0].masked_fill_(obs_ijk_fluid_im1jk, 0)

    jm1 = zero.where(j <= 0, j - 1)
    obst_ijm1k = zeroBy.where(j <= 0, (flags[idx_b, zero, k, jm1, i].eq(TypeObstacle))).__and__(mCont)
    U[:,1].masked_fill_(obst_ijm1k, 0)
    obs_ijk_fluid_ijm1k = zeroBy.where(j <= 0, (flags[idx_b, zero, k, jm1, i].eq(TypeFluid))).\
     __and__(cur_obs).__and__(mCont)
    U[:,1].masked_fill_(obs_ijk_fluid_ijm1k, 0)

    if (is3D):
        km1 = zero.where(k <= 0, k - 1)

        obst_ijkm1 = zeroBy.where(k <= 0, (flags[idx_b, zero, km1, j, i].eq(TypeObstacle))).__and__(mCont)
        U[:,2].masked_fill_(obst_ijkm1, 0)

        obs_ijk_fluid_ijkm1 = zeroBy.where(k <= 0, (flags[idx_b, zero, km1, j, i].eq(TypeFluid))). \
        _and__(cur_obs).__and__(mCont)
        U[:,2].masked_fill_(obs_ijk_fluid_ijkm1_, 0)

    # No-slip (aka stick) Boundary condition.
    # Normal AND tangential velocities are zero.
    # As the stick cells are also obstacle, we just need to add
    # tangential vel=0.
    # We work only on ghost cells (a velocity in obstacle cell) to enforce this condition.
    # For that reason, and needing access to multiple velocities, let's operate with float masks.

    ip1 = maxX.where(i>=(w-1), i + 1)
    jp1 = maxY.where(j>=(h-1), j + 1)

    cur_stick = cur_stick

    # Vertical velocities:
    fluid_im1jk = (flags[idx_b, zero, k, j, im1].eq(TypeFluid)).__and__(mCont)
    fluid_ip1jk = (flags[idx_b, zero, k, j, ip1].eq(TypeFluid)).__and__(mCont)

    v_im1jk = zeroF.where(i<=0, (U[idx_b, 1, k, j, im1]))
    v_ip1jk = zeroF.where(i>=(w-1), (U[idx_b, 1, k, j, ip1]))
    # Current cell is stick and left neighbor fluid. v(i,j) = -V(i-1,j)
    ghost_ijk_fluid_im1jk = cur_stick.__and__(fluid_im1jk).__and__(mCont)
    U[:,1].masked_scatter_(ghost_ijk_fluid_im1jk, \
            (-v_im1jk).masked_select(ghost_ijk_fluid_im1jk))
    # Current cell is stick and right neighbor fluid. v(i,j) = -V(i+1,j)
    ghost_ijk_fluid_ip1jk = cur_stick.__and__(fluid_ip1jk).__and__(mCont)
    U[:,1].masked_scatter_(ghost_ijk_fluid_ip1jk, \
            (-v_ip1jk).masked_select(ghost_ijk_fluid_ip1jk))
    # Both neighbors left and right are fluid. We approximate the ghost velocity as the mean between the two velocities.
    # This case should be avoided as much as possible (by putting walls of thickness 2 at least).
    ghost_ijk_fluid_imp1jk = cur_stick.__and__(fluid_im1jk).__and__(fluid_ip1jk).__and__(mCont)
    U[:,1].masked_scatter_(ghost_ijk_fluid_imp1jk, \
            ((0.5)*(-v_im1jk-v_ip1jk)).masked_select(ghost_ijk_fluid_imp1jk))

    # For horizontal velocities:
    fluid_ijm1k = (flags[idx_b, zero, k, jm1, i].eq(TypeFluid)).__and__(mCont)
    fluid_ijp1k = (flags[idx_b, zero, k, jp1, i].eq(TypeFluid)).__and__(mCont)

    u_ijm1k = zeroF.where(j<=0, (U[idx_b, 0, k, jm1, i]))
    u_ijp1k = zeroF.where(j>=(h-1), (U[idx_b, 0, k, jp1, i]))

    # Current cell is stick and bottom neighbor fluid. u(i,j) = -u(i,j-1)
    ghost_ijk_fluid_ijm1k = cur_stick.__and__(fluid_ijm1k).__and__(mCont)
    U[:,0].masked_scatter_(ghost_ijk_fluid_ijm1k, \
            (-u_ijm1k).masked_select(ghost_ijk_fluid_ijm1k))
    # Current cell is stick and upper neighbor fluid. u(i,j) = -u(i,j+1)
    ghost_ijk_fluid_ijp1k = cur_stick.__and__(fluid_ijp1k).__and__(mCont)
    U[:,0].masked_scatter_(ghost_ijk_fluid_ijp1k, \
            (-u_ijp1k).masked_select(ghost_ijk_fluid_ijp1k))
    # Both neighbors up and bottom are fluid. We approximate the ghost velocity as the mean between the two velocities.
    # This case should be avoided as much as possible (by putting walls of thickness 2 at least).
    ghost_ijk_fluid_ijmp1k = cur_stick.__and__(fluid_ijm1k).__and__(fluid_ijm1k).__and__(mCont)
    U[:,0].masked_scatter_(ghost_ijk_fluid_ijmp1k, \
            ((0.5)*(-u_ijm1k-u_ijp1k)).masked_select(ghost_ijk_fluid_ijmp1k))

    # Corner cases:
    cur_stick = cur_stick.float()
    left_stick = (flags_stick[idx_b, zero, k, j, im1].eq(TypeStick)).__and__(mCont)
    right_stick = (flags_stick[idx_b, zero, k, j, ip1].eq(TypeStick)).__and__(mCont)
    bottom_stick = (flags_stick[idx_b, zero, k, jm1, i].eq(TypeStick)).__and__(mCont)
    upper_stick = (flags_stick[idx_b, zero, k, jp1, i].eq(TypeStick)).__and__(mCont)

    # U velocity.
    G_U_ijk = (cur_stick.float() + left_stick.float() + bottom_stick.float() + \
                cur_stick.float() + left_stick.float() + upper_stick.float()).eq(3)
    U[:,0].masked_fill_(G_U_ijk, 0)

    # V velocity.
    G_V_ijk = (cur_stick.float() + left_stick.float() + bottom_stick.float() + \
                cur_stick.float() + right_stick.float() + bottom_stick.float()).eq(3)
    U[:,1].masked_fill_(G_V_ijk, 0)

