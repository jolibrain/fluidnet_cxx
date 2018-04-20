#include <iostream>
#include <sstream>
#include <mutex>

#include "grid.h"
#include "stack_trace.h"

GridBase::GridBase(at::Tensor* grid, bool is_3d) :
     is_3d_(is_3d), tensor_(grid), p_grid_(grid->data<float>()) {
  if (grid->ndimension() != 5) {
    AT_ERROR("GridBase: dim must be 5D (even is simulation is 2D).");
  }

  if (!is_3d_ && zsize() != 1) {
    AT_ERROR("GridBase: 2D grid must have zsize == 1.");
  }
}

float GridBase::getDx() const {
  const int64_t size_max = std::max(xsize(), std::max(ysize(), zsize()));
  return static_cast<float>(1) / static_cast<float>(size_max);
}

bool GridBase::isInBounds(const Int3& p, int bnd) const {
  bool ret = (p.x >= bnd && p.y >= bnd && p.x < xsize() - bnd &&
              p.y < ysize() - bnd);
  if (is_3d_) {
    ret &= (p.z >= bnd && p.z < zsize() - bnd);
  } else {
    ret &= (p.z == 0);
  }
  return ret;
}

bool GridBase::isInBounds(const vec3& p,
                                    int bnd) const {
  return isInBounds(toInt3(p), bnd);
}

int64_t GridBase::index5d(int64_t i, int64_t j, int64_t k,
                                    int64_t c, int64_t b) const {
  if (i >= xsize() || j >= ysize() || k >= zsize() || c >= nchan() ||
      b >= nbatch() || i < 0 || j < 0 || k < 0 || c < 0 || b < 0) {
    std::cout << "Error index5D out of bounds" << std::endl << std::flush;
    std::lock_guard<std::mutex> lock(mutex_);
    std::stringstream ss;
    ss << "GridBase: index5d out of bounds:" << std::endl
       << "  (i, j, k, c, b) = (" << i << ", " << j
       << ", " << k << ", " << c << ", " << b << "), size = (" << xsize()
       << ", " << ysize() << ", " << zsize() << ", " << nchan()
       << nbatch() << ")";
    std::cerr << ss.str() << std::endl << "Stack trace:" << std::endl;
    PrintStacktrace();
    std::cerr << std::endl;
    AT_ERROR("GridBase: index4d out of bounds");
    return 0;
  }
  return (i * xstride() + j * ystride() + k * zstride() + c * cstride() +
          b * bstride());
}
void GridBase::buildIndex(
    int64_t& xi, int64_t& yi, int64_t& zi, float& s0, float& t0, float& f0,
    float& s1, float& t1, float& f1, const vec3& pos) const {
  const float px = pos.x - static_cast<float>(0.5);
  const float py = pos.y - static_cast<float>(0.5);
  const float pz = pos.z - static_cast<float>(0.5);
  xi = static_cast<int64_t>(px);
  yi = static_cast<int64_t>(py);
  zi = static_cast<int64_t>(pz);
  s1 = px - static_cast<float>(xi);
  s0 = static_cast<float>(1) - s1;
  t1 = py - static_cast<float>(yi);
  t0 = static_cast<float>(1) - t1;
  f1 = pz - static_cast<float>(zi);
  f0 = static_cast<float>(1) - f1;
  // Clamp to border.
  if (px < static_cast<float>(0)) {
    xi = 0;
    s0 = static_cast<float>(1);
    s1 = static_cast<float>(0);
  }
  if (py < static_cast<float>(0)) {
    yi = 0;
    t0 = static_cast<float>(1);
    t1 = static_cast<float>(0);
  }
  if (pz < static_cast<float>(0)) {
    zi = 0;
    f0 = static_cast<float>(1);
    f1 = static_cast<float>(0);
  }
  if (xi >= xsize() - 1) {
    xi = xsize() - 2;
    s0 = static_cast<float>(0);
    s1 = static_cast<float>(1);
  }
  if (yi >= ysize() - 1) {
    yi = ysize() - 2;
    t0 = static_cast<float>(0);
    t1 = static_cast<float>(1);
  }
  if (zsize() > 1) {
    if (zi >= zsize() - 1) {
    }
  } 
}

std::mutex GridBase::mutex_;

