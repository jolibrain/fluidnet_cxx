import torch

def flagsToOccupancy(flags):
    TypeFluid = 1
    TypeObstacle = 2
    occupancy = flags.clone()
    flagsFluid = occupancy.eq(TypeFluid)
    flagsObstacle = occupancy.eq(TypeObstacle)
    occupancy.masked_fill_(flagsFluid, 0)
    occupancy.masked_fill_(flagsObstacle, 1)
    return occupancy
