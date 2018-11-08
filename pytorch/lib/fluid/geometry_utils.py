import torch
from . import CellType

def createCylinder(batch_dict, centerX, centerY, radius):
    r"""Adds (inplace) a cylinder to the flags tensor in input batch.

    Arguments:
        batch_dict (dict): Input batch of tensor.
        centerX (float): X-coordinate of cylinder center.
        centerY (float): Y-coordinate of cylinder center.
        radius (float): Radius of cylinder.

    """
    cuda = torch.device('cuda')
    assert 'flags' in batch_dict, 'Error: flags key is not in batch dict'
    flags = batch_dict['flags']
    assert flags.dim() == 5, 'Input flags must have 5 dimensions'
    assert flags.size(0) == 1, 'Only batches of size 1 allowed (inference)'
    xdim = flags.size(4)
    ydim = flags.size(3)
    zdim = flags.size(2)
    is3D = (zdim > 1)

    # Create the cylinder
    X = torch.arange(0, xdim, device=cuda).view(xdim).expand((1,ydim,xdim))
    Y = torch.arange(0, ydim, device=cuda).view(ydim, 1).expand((1,ydim,xdim))

    dist_from_center = (X - centerX).pow(2) + (Y-centerY).pow(2)
    mask_cylinder = dist_from_center <= radius*radius

    flags.masked_fill_(mask_cylinder, CellType.TypeObstacle)
    batch_dict['flags'] = flags

def createBox2D(batch_dict, x0, x1, y0, y1):
    r"""Adds (inplace) a 2D Box to the flags tensor in input batch.

    Arguments:
        batch_dict (dict): Input batch of tensor.
        x0 (float): bottom-left x-coordinate.
        y0 (float): bottom-left y-coordinate.
        x1 (float): upper-right x-coordinate.
        y1 (float): upper-right y-coordinate.

    """
    cuda = torch.device('cuda')
    assert 'flags' in batch_dict, 'Error: flags key is not in batch dict'
    flags = batch_dict['flags']
    assert flags.dim() == 5, 'Input flags must have 5 dimensions'
    assert flags.size(0) == 1, 'Only batches of size 1 allowed (inference)'
    xdim = flags.size(4)
    ydim = flags.size(3)
    zdim = flags.size(2)
    is3D = (zdim > 1)

    # Create the cylinder
    X = torch.arange(0, xdim, device=cuda).view(xdim).expand((1,ydim,xdim))
    Y = torch.arange(0, ydim, device=cuda).view(ydim, 1).expand((1,ydim,xdim))

    mask_box_2D = (X >= x0).__and__(X < x1).__and__\
                (Y >= y1).__and__(Y < y1)

    flags.masked_fill_(mask_box2_2D, CellType.TypeObstacle)
    batch_dict['flags'] = flags

