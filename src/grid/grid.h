#pragma once

#include <iostream>
#include <sstream>
#include "ATen/ATen.h"
#include "int3.h"

class GridBase {
public:
  explicit GridBase(at::Tensor* grid, bool is_3d);

  int32_t nbatch() const { return tensor_->size[0]; }
  int32_t nchan() const { return tensor_->size[1]; }
  int32_t zsize() const { return tensor_->size[2]; }
  int32_t ysize() const { return tensor_->size[3]; }
  int32_t xsize() const { return tensor_->size[4]; }

  int32_t bstride() const { return tensor_->stride[0]; }
  int32_t cstride() const { return tensor_->stride[1]; }
  int32_t zstride() const { return tensor_->stride[2]; }
  int32_t ystride() const { return tensor_->stride[3]; }
  int32_t xstride() const { return tensor_->stride[4]; }

private:
  at::Tensor* const tensor_;
  float* const p_grid_;
  const bool is_3d_; 


};
