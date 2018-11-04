from enum import IntEnum

# We use the same convention as Mantaflow and FluidNet

class CellType(IntEnum):
    TypeNone = 0
    TypeFluid = 1
    TypeObstacle = 2
    TypeEmpty = 4
    TypeInflow = 8
    TypeOutflow = 16
    TypeOpen = 32
    TypeStick = 128
    TypeReserved = 256

