#pragma once

#include "torch/torch.h"
#include "cell_type.h"

// See the .cpp file for detailed comments.
namespace fluid {

typedef at::Tensor T;

void getPixelCenter(const T& pos, T& ix);

T isOutOfDomain(const T& pos, const T& flags);

T isBlockedCell(const T& pos, const T& flags);

void clampToDomain(T& pos, const T& flags);

typedef enum Quadrants {
  RIGHT = 0,
  LEFT = 1,
  MIDDLE = 2
} Quadrants;

T HitBoundingBox(const T& minB, const T& maxB,
                 const T& origin, const T& dir,
                 const T& mask, T& coord);

T calcRayBoxIntersection(const T& pos, const T& dt, const T& ctr,
                         const float hit_margin, const T& mask,  T& ipos);

T calcRayBorderIntersection(const T& pos, const T& next_pos,
                            const T& flags, const float hit_margin,
                            const T& mOutDom,
                            T& ipos);

void calcLineTrace(const T& pos, const T& delta, const T& flags,
                T& new_pos, const bool do_line_trace);

} // namespace fluid
