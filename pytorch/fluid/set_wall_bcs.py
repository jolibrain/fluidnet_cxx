import torch

def setWallBcs(U, flags):

    assert (U.dim() == 5 and flags.dim() == 5), 'Dimension mismatch'
    assert flags.size(1) == 1, 'flags is not a scalar'
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
    assert (U.is_contiguous() and flags.is_contiguous()), 'Input is not contiguous'

    i = torch.arange(0, w).view(1,w).expand(bsz, d, h, w).type(torch.LongTensor)
    j = torch.arange(0, h).view(1,h,1).expand(bsz, d, h, w).type(torch.LongTensor)
    if (is3D):
        k = torch.arange(0, d).view(1,d,1,1).expand(bsz, d, h, w).type(torch.LongTensor)

    zero = torch.zeros_like(i)
    zeroBy = zero.type(torch.ByteTensor)

    idx_b = torch.arange(0, bsz).view(bsz, 1, 1, 1).expand(bsz,d,h,w).type(torch.LongTensor)

    mCont = torch.ones_like(zeroBy)

    cur_fluid = flags.eq(1).squeeze(1)
    cur_obs = flags.eq(2).squeeze(1)
    mNotFluidNotObs = cur_fluid.ne(1).__and__(cur_obs.ne(1))
    mCont.masked_fill_(mNotFluidNotObs, 0)

    i_l = zero.where( (i <=0), i - 1)
    obst100 = zeroBy.where( i <= 0, (flags.index(idx_b, zero, k, j, i_l).eq(2))).__and__(mCont)
    U.select(1,0).masked_fill_(obst100, 0)

    obs_fluid100 = zeroBy.where( i <= 0, (flags.index(idx_b, zero, k, j, i_l).eq(1))). \
     __and__(cur_obs).__and__(mCont)
    U.select(1,0).masked_fill_(obs_fluid100, 0)

    j_l = zero.where( (j <= 0), j - 1)
    obst010 = zeroBy.where( j <= 0, (flags.index(idx_b, zero, k, j_l, i).eq(2))).__and__(mCont);
    U.select(1,1).masked_fill_(obst010, 0)
    obs_fluid010 = zeroBy.where( j <= 0, (flags.index(idx_b, zero, k, j_l, i).eq(1))).\
     __and__(cur_obs).__and__(mCont)
    U.select(1,1).masked_fill_(obs_fluid010, 0)

    if (is3D):
        k_l = zero.where( (k <= 0), k - 1)

        obst001 = zeroBy.where( k <= 0, (flags.index(idx_b, zero, k_l, j, i).eq(2))).__and__(mCont)
        U.select(1,2).masked_fill_(obst001, 0)

        obs_fluid001 = zeroBy.where( k <= 0, (flags.index(idx_b, zero, k_l, j, i).eq(1))). \
        _and__(cur_obs).__and__(mCont)
        U.select(1,2).masked_fill_(obs_fluid001, 0)



U = torch.rand(1, 2, 1, 5, 5)
flags = torch.IntTensor(1,1,1,5,5).random_(1,3)

setWallBcs(U, flags)
