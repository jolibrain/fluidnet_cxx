from .grid import getDx
from .set_wall_bcs import setWallBcs
from .flags_to_occupancy import flagsToOccupancy
from .velocity_divergence import velocityDivergence
from .velocity_update import velocityUpdate
from .util import emptyDomain
from .cpp.advection import advectScalar, advectVelocity
from .cpp.solve_linear_sys import solveLinearSystemJacobi

