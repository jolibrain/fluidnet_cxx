import torch

from . import CellType

def emptyDomain(flags, boundary_width = 1):
    cuda = torch.device('cuda')
    assert boundary_width > 0, 'Boundary width must be greater than zero!'
    assert flags.dim() == 5, 'Flags tensor should be 5D'
    assert flags.size(1) == 1, 'Flags should have only one channels (scalar field)'

    is3D = flags.size(2) > 1

    assert ( ((not is3D) or (flags.size(2) > boundary_width*2)) and \
            (flags.size(3) > boundary_width*2) or (flags.size(4) > boundary_width*2)), \
            'Simulation domain is not big enough'
    xdim  = flags.size(4)
    ydim  = flags.size(3)
    zdim  = flags.size(2)
    bnd = boundary_width

    index_x = torch.arange(0, xdim, device=cuda).view(xdim).expand_as(flags[0][0])
    index_y = torch.arange(0, ydim, device=cuda).view(ydim, 1).expand_as(flags[0][0])
    if (is3D):
        index_z = torch.arange(0, zdim, device=cuda).view(zdim, 1 , 1).expand_as(flags[0][0])

    if (not is3D):
        index_ten = torch.stack((index_x, index_y), dim=0)
    else:
        index_ten = torch.stack((index_x, index_y, index_z), dim=0)

    maskBorder = (index_ten.select(0,0) < bnd).__or__ \
                            (index_ten.select(0,0) > xdim - 1 - bnd).__or__\
                            (index_ten.select(0,1) < bnd).__or__\
                            (index_ten.select(0,1) > ydim - 1 - bnd)
    if (is3D):
        maskBorder = maskBorder.__or__(index_ten.select(0,2) < bnd).__or__\
                                      (index_ten.select(0,2) > zdim - 1 - bnd)

    flags.masked_fill_(maskBorder, CellType.TypeObstacle)
    flags.masked_fill_((maskBorder == 0), CellType.TypeFluid)


