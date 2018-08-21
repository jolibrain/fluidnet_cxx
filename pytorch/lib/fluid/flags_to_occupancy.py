import torch
from . import CellType

def flagsToOccupancy(flags):
    occupancy = flags.clone()
    flagsFluid = occupancy.eq(CellType.TypeFluid)
    flagsObstacle = occupancy.eq(CellType.TypeObstacle)
    occupancy.masked_fill_(flagsFluid, 0)
    occupancy.masked_fill_(flagsObstacle, 1)
    return occupancy

