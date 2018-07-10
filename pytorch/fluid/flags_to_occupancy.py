import torch

def FlagsToOccupancy(flags):
    occupancy = flags.clone()
    flagsFluid = occupancy.eq(1)
    flagsObstacle = occupancy.eq(2)
    occupancy.masked_fill_(flagsFluid, 0)
    occupancy.masked_fill_(flagsObstacle, 1)
    return occupancy
