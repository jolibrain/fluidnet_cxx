#include <iostream>
#include <sstream>

#include "grid.h"

GridBase::GridBase(at::Tensor* grid, bool is_3d) :
     is_3d_(is_3d), tensor_(grid), p_grid_(grid.data<float>()) {
  if (grid->nDimension != 5) {
    AT_ERROR("GridBase: dim must be 5D (even is simulation is 2D).");
  }

  if (!is_3d_ && zsize() != 1) {
    AT_ERROR("GridBase: 2D grid must have zsize == 1.");
  }
}
