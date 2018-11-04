import torch
from . import CellType



def flagsToOccupancy(flags):
    r"""Transforms the flags tensor to occupancy tensor (0 for fluids, 1 for obstacles).

    Arguments:
        flags (Tensor): Input occupancy grid.
    Output:
        occupancy (Tensor): Output occupancy grid (0s and 1s).
    """
    occupancy = flags.clone()
    flagsFluid = occupancy.eq(CellType.TypeFluid)
    flagsObstacle = occupancy.eq(CellType.TypeObstacle)
    occupancy.masked_fill_(flagsFluid, 0)
    occupancy.masked_fill_(flagsObstacle, 1)
    return occupancy

