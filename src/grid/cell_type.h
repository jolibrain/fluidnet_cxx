#pragma once

// These are the same enum values used in Manta. We can't include grid.h
// from Manta without pulling in the entire library, so we'll just redefine
// them here.
enum CellType {
    TypeNone = 0,
    TypeFluid = 1,
    TypeObstacle = 2,
    TypeEmpty = 4,
    TypeInflow = 8,
    TypeOutflow = 16,
    TypeOpen = 32,
    TypeStick = 128,
    TypeReserved = 256,
    TypeZeroPressure = (1<<15)
};

