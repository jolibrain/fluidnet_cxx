#pragma once

#include <iostream>
#include <sstream>
#include <mutex>

#include "ATen/ATen.h"
#include "int3.h"
#include "vec3.h"

class GridBase {
public:
  explicit GridBase(at::Tensor* grid, bool is_3d);

  int64_t nbatch() const { return tensor_->size(0); }
  int64_t nchan() const { return tensor_->size(1); }
  int64_t zsize() const { return tensor_->size(2); }
  int64_t ysize() const { return tensor_->size(3); }
  int64_t xsize() const { return tensor_->size(4); }

  int64_t bstride() const { return tensor_->stride(0); }
  int64_t cstride() const { return tensor_->stride(1); }
  int64_t zstride() const { return tensor_->stride(2); }
  int64_t ystride() const { return tensor_->stride(3); }
  int64_t xstride() const { return tensor_->stride(4); }

  bool is_3d() const { return is_3d_; }
  Int3 getSize() const { return Int3(xsize(), ysize(), zsize()); }

  float getDx() const;

  bool isInBounds(const Int3& p, int bnd) const;

  bool isInBounds(const vec3& p, int bnd) const;

private:
  at::Tensor* const tensor_;
  float* const p_grid_;
  const bool is_3d_; 
  static std::mutex mutex_;

  // The indices i, j, k, c, b are x, y, z, chan and batch respectively.
  int64_t index5d(int64_t i, int64_t j, int64_t k, int64_t c, int64_t b) const;

protected:
  // Use operator() methods in child classes to get at data.
  // Note: if the storage is offset (i.e. because we've selected along the
  // batch dim), this is taken care of in  at::Tensor.data()(i.e. it returns
  // self->storage->data + self->storageOffset).
  float& data(int64_t i, int64_t j, int64_t k, int64_t c, int64_t b) {
    return p_grid_[index5d(i, j, k, c, b)];
  }

  float data(int64_t i, int64_t j, int64_t k, int64_t c, int64_t b) const {
    return p_grid_[index5d(i, j, k, c, b)];
  }

  float& data(const Int3& pos, int64_t c, int64_t b) {
    return data(pos.x, pos.y, pos.z, c, b);
  }

  float data(const Int3& pos, int64_t c, int64_t b) const {
    return data(pos.x, pos.y, pos.z, c, b);
  }

  void buildIndex(int64_t& xi, int64_t& yi, int64_t& zi,
                  float& s0, float& t0, float& f0,
                  float& s1, float& t1, float& f1,
                  const vec3& pos) const;
};
