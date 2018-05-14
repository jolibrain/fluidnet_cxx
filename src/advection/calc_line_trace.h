#pragma once

#include "grid/grid.h"

namespace fluid { 
  void GetPixelCenter(const T& pos, T& ix);
  
  bool IsOutOfDomainReal(const T& pos, const FlagGrid& flags);
  
  bool IsBlockedCell(const FlagGrid& flags,
                     const T& pos, int32_t b);
  
  void ClampToDomainReal(T& pos, const FlagGrid& flags);
  
  bool IsBlockedCellReal(const FlagGrid& flags,
                         const T& pos, T b);
  
  bool HitBoundingBox(const T* minB, const T* maxB, 
                      const T* origin, const T* dir,
                      T* coord);
  
  bool calcRayBoxIntersection(const T& pos,
                              const T& dt,
                              const T& ctr,
                              const T hit_margin, T* ipos);
  
  bool calcRayBorderIntersection(const T& pos,
                                 const T& next_pos,
                                 const FlagGrid& flags,
                                 const T hit_margin,
                                 T* ipos);
  
  bool calcLineTrace(const T& pos, const T& delta,
                     const FlagGrid& flags, const T ibatch,
                     T* new_pos, const bool do_line_trace);
  
} // namespace fluid 
