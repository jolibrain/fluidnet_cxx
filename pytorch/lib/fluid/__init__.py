from .cell_type import CellType
from .grid import getDx, getCentered
from .set_wall_bcs import setWallBcs
from .set_wall_bcs_stick import setWallBcsStick
from .flags_to_occupancy import flagsToOccupancy
from .velocity_divergence import velocityDivergence
from .velocity_update import velocityUpdate
from .source_terms import addBuoyancy, addGravity
from .viscosity import addViscosity
from .util import emptyDomain
from .cpp.advection import advectScalar, advectVelocity
from .cpp.solve_linear_sys import solveLinearSystemJacobi

