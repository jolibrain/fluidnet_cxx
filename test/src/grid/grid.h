#pragma once

#include <iostream>
#include <sstream>
#include "ATen/ATen.h"
#include "ATen/Error.h"
#include "int3.h"

class GridBase {
public:
  explicit GridBase(at::Tensor* grid, bool is_3d);

  int32_t nbatch() const { return tensor_->size(0); }
  int32_t nchan() const { return tensor_->size(1); }
  int32_t zsize() const { return tensor_->size(2); }
  int32_t ysize() const { return tensor_->size(3); }
  int32_t xsize() const { return tensor_->size(4); }

  int32_t bstride() const { return tensor_->stride(0); }
  int32_t cstride() const { return tensor_->stride(1); }
  int32_t zstride() const { return tensor_->stride(2); }
  int32_t ystride() const { return tensor_->stride(3); }
  int32_t xstride() const { return tensor_->stride(4); }

private:
  at::Tensor* const tensor_;
  float* const p_grid_;
  const bool is_3d_; 


};

GridBase::GridBase(at::Tensor* grid, bool is_3d) :
     is_3d_(is_3d), tensor_(grid), p_grid_(grid->data<float>()) {
  if (grid->ndimension() != 5) {
    AT_ERROR("GridBase: dim must be 5D (even is simulation is 2D).");
  }

  if (!is_3d_ && zsize() != 1) {
    AT_ERROR("GridBase: 2D grid must have zsize == 1.");
  }
}

