#include <assert.h>
#include <limits>
#include "ATen/ATen.h"
#include "calc_line_trace.h"

namespace fluid {  
  
  // We never want positions to go exactly to the border or exactly to the edge
  // of an occupied piece of geometry. Therefore all rays will be truncated by
  // a very small amount (hit_margin).
   
  #define HIT 1e-5
  #define EPS 1e-12

  // Get the integer index of the current voxel.
  // Manta defines 0.5 as the center of the first cell, you can see this in
  // manta/source/grid.h Grid::getInterpolated() and the lower level call in
  // manta/source/util/interpol.h interpol(), where the input position has a
  // pos - 0.5 applied to it (our interpol function does this as well).
  void GetPixelCenter(const T& pos, T& ix) {
    // Note: you could either calculate (int)round(pos.x - 0.5), or you can
    // just round down without taking off the 0.5 value.
  
    ix = pos.toType(getType(pos.type().backend(), at::kInt));
  }
  
  // Note: IsOutOfDomainReal considers AGAINST the domain to be out of domain.
  // It also considers the space from the left of the first cell to the center
  // (even though we don't have samples there) and the space from the right of the
  // last cell to the border.
  bool IsOutOfDomainReal(
      const T& pos, const FlagGrid& flags) {
    return (toBool((pos[0] <= 0).__or__  // LHS of cell.
                  (pos[0] >= flags.xsize()).__or__  // RHS of cell.
                  (pos[1] <= 0).__or__
                  (pos[1] >= flags.ysize()).__or__
                  (pos[2] <= 0).__or__
                  (pos[2] >= flags.zsize()) ) );
  }
  
  bool IsBlockedCell(const FlagGrid& flags,
                                   const T& pos, T b) {
    // Returns true if the cell is blocked.
    // Shouldn't be called on point outside the domain.
    if (flags.isOutOfDomain(pos[0], pos[1], pos[2], b)) {
      // Hard assert here.
      AT_ERROR("ERROR: IsBlockedCell called on out of domain coords.");
    }
    return !flags.isFluid(pos, b);
  }
  
  void ClampToDomainReal(
      T& pos, const FlagGrid& flags) {
    // Clamp to a position epsilon inside the simulation domain. 
    const T hit_margin = at::infer_type(pos).scalarTensor(HIT);
    pos = at::min(at::max(pos, hit_margin), 
                          flags.xsize() - hit_margin);
  }
  
  // This takes in the float position, calculates the current voxel index
  // and performs the integer lookup on that.
  bool IsBlockedCellReal(const FlagGrid& flags,
                                       const T& pos, T b) {
    T ix = getType(pos.type().backend(),at::kInt).zeros({3});
    GetPixelCenter(pos, ix);
    return IsBlockedCell(flags, ix, b); 
  }

  typedef enum Quadrants {
    RIGHT = 0,
    LEFT = 1,
    MIDDLE = 2
  } Quadrants;
  
  // https://github.com/erich666/GraphicsGems/blob/master/gems/RayBox.c
  // And modified it (there were actually a few numerical precision bugs).
  // I tested the hell out of it, so it seems to work.
  //
  // @param hit_margin ue >= 0 describing margin added to hit to
  // prevent interpenetration.
  bool HitBoundingBox(const T* minB, const T* maxB,  // box
                      const T* origin, const T* dir,  // ray
                      T* coord) {  // hit point.
    const T hit_margin = at::infer_type(*coord).scalarTensor(HIT);
    const T epsilon = at::infer_type(*coord).scalarTensor(EPS);
    
    char inside = true;
    Quadrants quadrant[3];
    register int i;
    int whichPlane;
    T maxT = at::infer_type(*minB).zeros({3});
    T candidate_plane = at::infer_type(*minB).zeros({3});
    
  
    // Find candidate planes; this loop can be avoided if rays cast all from the
    // eye (assume perpsective view).
    for (i = 0; i < 3; i++) {
      if (toBool( (*origin)[i] < (*minB)[i]) ) {
        quadrant[i] = LEFT;
        candidate_plane[i] = (*minB)[i];
        inside = false;
      } else if (toBool( (*origin)[i] > (*maxB)[i]) ) {
        quadrant[i] = RIGHT;
        candidate_plane[i] = (*maxB)[i];
        inside = false;
      } else {
        quadrant[i] = MIDDLE;
      }
    }
  
    // Ray origin inside bounding box.
    if (inside) {
      (*coord) = (*origin);    
      return true;
    }
  
    // Calculate T distances to candidate planes.
    for (i = 0; i < 3; i++) {
      if (quadrant[i] != MIDDLE && toBool( (*dir)[i] != 0.0) ) {
        maxT[i] = (candidate_plane[i] - (*origin)[i]) / (*dir)[i];
      } else {
        maxT[i] = -1.0;
      }
    }
  
    // Get largest of the maxT's for final choice of intersection.
    whichPlane = 0;
    for (i = 1; i < 3; i++) {
      if (toBool(maxT[whichPlane] < maxT[i]) ) {
        whichPlane = i;
      }
    }
  
    // Check final candidate actually inside box and calculate the coords (if
    // not).
    if (toBool(maxT[whichPlane] < (0.0)) ) {
      return false;
    }
  
    const T err_tol = at::infer_type(maxT).scalarTensor(1e-6);
    for (i = 0; i < 3; i++) {
      if (whichPlane != i) {
        (*coord)[i] = (*origin)[i] + maxT[whichPlane] * (*dir)[i];
        if (toBool( ((*coord)[i] < ( (*minB)[i] - err_tol))
        .__or__( (*coord)[i] > ( (*maxB)[i] + err_tol)) ) ){
          return false;
        }
      } else {
        (*coord)[i] = candidate_plane[i];
      }
    }
  
    return true;
  }   
  
  // calcRayBoxIntersection will calculate the intersection point for the ray
  // starting at pos, and pointing along dt (which should be unit length).
  // The box is size 1 and is centered at ctr.
  bool calcRayBoxIntersection(const T& pos,
                              const T& dt,
                              const T& ctr, 
                              const T hit_margin, T* ipos) {
    const T epsilon = at::infer_type(pos).scalarTensor(EPS);

    if (toBool(hit_margin < 0)) {
      AT_ERROR("Error: hit_margin < 0");
    }
    T box_min = at::infer_type(pos).zeros({3});
    box_min = ctr - 0.5 - hit_margin;
    T box_max = at::infer_type(pos).zeros({3});
    box_max = ctr + 0.5 + hit_margin;
  
    bool hit = HitBoundingBox(&box_min, &box_max,  // box
                              &pos, &dt,  // ray
                              ipos);
    return hit;
  }
  
  // calcRayBorderIntersection will calculate the intersection point for the ray
  // starting at pos and pointing to next_pos.
  //
  // IMPORTANT: This function ASSUMES that the ray actually intersects. Nasty
  // things will happen if it does not.
  // EDIT(tompson, 09/25/16): This is so important that we'll actually double
  // check the input coords anyway.
  bool calcRayBorderIntersection(const T& pos,
                                 const T& next_pos,
                                 const FlagGrid& flags,
                                 const T hit_margin,
                                 T* ipos) {
    const T epsilon = at::infer_type(pos).scalarTensor(EPS);
    if (toBool(hit_margin <= 0)) {
      AT_ERROR("Error: calcRayBorderIntersection hit_margin < 0.");
    }
  
    // The source location should be INSIDE the boundary.
    if (IsOutOfDomainReal(pos, flags)) {
      AT_ERROR("Error: source location is already outside the domain!");
    }
    // The target location should be OUTSIDE the boundary.
    if (!IsOutOfDomainReal(next_pos, flags)) {
      AT_ERROR("Error: target location is already outside the domain!");
    }
  
    // Calculate the minimum step length to exit each face and then step that
    // far. The line equation is:
    //   P = gamma * (next_pos - pos) + pos.
    // So calculate gamma required to make P < + margin for each dim
    // independently.
    //   P_i = m --> m - pos_i = gamma * (next_pos_i - pos_i)
    //   --> gamma_i = (m - pos_i) / (next_pos_i - pos_i)
    T min_step = at::infer_type(pos).scalarTensor(std::numeric_limits<float>::max());
    if (toBool(next_pos[0] <= hit_margin)) {  // left face.
      const T dx = next_pos[0] - pos[0];
      if (toBool(at::abs(dx) >= epsilon)) {
        const T xstep = (hit_margin - pos[0]) / dx;
        min_step = at::min(min_step, xstep);
      }
    }
    if (toBool(next_pos[1] <= hit_margin)) {
      const T dy = next_pos[1] - pos[1];
      if (toBool(at::abs(dy) >= epsilon)) {
        const T ystep = (hit_margin - pos[1]) / dy;
        min_step = at::min(min_step, ystep);
      }
    }
    if (toBool(next_pos[2] <= hit_margin)) {
      const T dz = next_pos[2] - pos[2];
      if (toBool(at::abs(dz) >= epsilon)) {
        const T zstep = (hit_margin - pos[2]) / dz;
        min_step = at::min(min_step, zstep);
      }
    }
    // Also calculate the min step to exit a positive face.
    //   P_i = dim - m --> dim - m - pos_i = gamma * (next_pos_i - pos_i)
    //   --> gamma = (dim - m - pos_i) / (next_pos_i - pos_i)
    if (toBool(next_pos[0] >=
       (at::infer_type(next_pos).scalarTensor(flags.xsize())
        - hit_margin)) ) {  // right face.
      const T dx = next_pos[0] - pos[0];
      if (toBool(at::abs(dx) >= epsilon) ) {
        const T xstep = (at::infer_type(next_pos).scalarTensor(flags.xsize())
        - hit_margin - pos[0]) / dx;
        min_step = at::min(min_step, xstep);
      }
    }
    if (toBool(next_pos[1] >=
       (at::infer_type(next_pos).scalarTensor(flags.ysize())
       - hit_margin)) ) {  
      const T dy = next_pos[1] - pos[1];
      if (toBool(at::abs(dy) >= epsilon)) {
        const T ystep = (at::infer_type(next_pos).scalarTensor(flags.ysize())
        - hit_margin - pos[0]) / dy;
        min_step = at::min(min_step, ystep);
      }
    }
    if (toBool(next_pos[2] >=
       (at::infer_type(next_pos).scalarTensor(flags.zsize())
       - hit_margin)) ) {  
      const T dz = next_pos[2] - pos[2];
      if (toBool(at::abs(dz) >= epsilon)) {
        const T zstep = (at::infer_type(next_pos).scalarTensor(flags.zsize())
        - hit_margin - pos[2]) / dz;
        min_step = at::min(min_step, zstep);
      }
    }

    if (toBool( (min_step < 0)
       .__or__(min_step >= at::infer_type(min_step)
       .scalarTensor(std::numeric_limits<float>::max()))) ) {
      return false;
    }
  
    // Take the minimum step.
    (*ipos) = min_step * (next_pos - pos) + pos;
  
    return true;
  }
  
  // The following function performs a line trace along the displacement vector
  // and returns either:
  // a) The position 'p + delta' if NO geometry is found on the line trace. or
  // b) The position at the first geometry blocker along the path.
  // The search is exhaustive (i.e. O(n) in the length of the displacement vector)
  //
  // Note: the returned position is NEVER in geometry or outside the bounds. We
  // go to great lengths to ensure this.
  //
  // TODO(tompsion): This is probably not efficient at all.
  // It also has the potential to miss geometry along the path if the width
  // of the geometry is less than 1 grid.
  //
  // For float grids values are stored at i+0.5, j+0.5, k+0.5. i.e. the center of
  // the first cell is (0.5, 0.5, 0.5) so the corner is (0, 0, 0). Likewise the
  // center of the last cell is (xsize - 1 + 0.5, ...) so the corner is
  // (xsize, ysize, zsize).
  //
  // For MAC grids values are stored at i, j+0.5, k+0.5 for the x component.
  // So the MAC component for the (i, j, k) index is on the left, bottom and back
  // faces of the cell respectively (i.e. the negative edge).
  //
  // So, if you want to START a line trace at the index (i, j, k) you should add
  // 0.5 to each component before calling this function as (i, j, k) converted to
  // float will actually be the (left, bottom, back) side of that cell.
  bool calcLineTrace(const T& pos, const T& delta,
                     const FlagGrid& flags, const T ibatch,
                     T* new_pos, const bool do_line_trace) {
    const T hit_margin = at::infer_type(pos).scalarTensor(HIT);
    const T epsilon = at::infer_type(pos).scalarTensor(EPS);
    // We can choose to not do a line trace at all.
    if (!do_line_trace) {
      (*new_pos) = pos + delta;
      return false;
    }
  
    // If we're ALREADY in a obstacle segment (or outside the domain) then a lot
    // of logic below will fail. This function should only be called on fluid
    // cells!
    if (IsOutOfDomainReal(pos, flags)) {
      AT_ERROR("Error: CalcLineTrace was called on a out of domain cell!");
    }
    if (IsBlockedCellReal(flags, pos, ibatch)) {
      AT_ERROR("Error: CalcLineTrace was called on a non-fluid cell!");
    }
  
    (*new_pos) = pos;
  
    const T length = delta.norm();
    if (toBool(length <= epsilon)) {
      // We're not being asked to step anywhere. Return false and copy the pos.
      // (copy already done above).
      return false;
    }
    // Figure out the step size in x, y and z for our marching.
    T dt = delta / length;
  
    // Otherwise, we start the line search, by stepping a unit length along the
    // vector and checking the neighbours.
    //
    // A few words about the implementation (because it's complicated and perhaps
    // needlessly so). We maintain a VERY important loop invariant: new_pos is
    // NEVER allowed to enter solid geometry or go off the domain. next_pos
    // is the next step's tentative location, and we will always try and back
    // it off to the closest non-geometry valid cell before updating new_pos.
    //
    // We will also go to great lengths to ensure this loop invariant is
    // correct (probably at the expense of speed).
    T cur_length = at::infer_type(dt).scalarTensor(0);
    T next_pos = at::infer_type(pos).zeros({3});  // Tentative step location.
    while (toBool(cur_length < (length - hit_margin))) {
      // We haven't stepped far enough. So take a step.
      T cur_step = at::min(length - cur_length,
                           at::infer_type(pos).scalarTensor(1));
      next_pos = (*new_pos) + (dt * cur_step);
  
      // Check to see if we went too far.
      // TODO(tompson): This is not correct, we might skip over small
      // pieces of geometry if the ray brushes against the corner of a
      // occupied voxel, but doesn't land in it. Fix this (it's very rare though).
    
      // There are two possible cases. We've either stepped out of the domain
      // or entered a blocked cell.
      if (IsOutOfDomainReal(next_pos, flags)) {
        // Case 1. 'next_pos' exits the grid.
        T ipos = at::infer_type(pos).zeros({3});
        const bool hit = calcRayBorderIntersection(
            *new_pos, next_pos, flags, hit_margin, &ipos);
        if (!hit) {
          // This is an EXTREMELY rare case. It happens because either the ray is
          // almost parallel to the domain boundary, OR floating point round-off
          // causes the intersection test to fail.
          
          // In this case, fall back to simply clamping next_pos inside the domain
          // boundary. It's not ideal, but better than a hard failure (the reason
          // why it's wrong is that clamping will bring the point off the ray).
          ipos = next_pos;
          ClampToDomainReal(ipos, flags);
        }
  
        // Do some sanity checks. I'd rather be slow and correct...
        // The logic above should always put ipos back hit_margin inside the
        // simulation domain.
        if (IsOutOfDomainReal(ipos, flags)) {
          AT_ERROR("Error: case 1 exited bounds!");
        }
  
        if (!IsBlockedCellReal(flags, ipos, ibatch)) {
          // OK to return here (i.e. we're up against the border and not
          // in a blocked cell).
          (*new_pos) = ipos;
          return true;
        } else {
          // Otherwise, we hit the border boundary, but we entered a blocked cell.
          // Continue on to case 2.
          next_pos = ipos;
        }
      }
      if (IsBlockedCellReal(flags, next_pos, ibatch)) {
        // Case 2. next_pos enters a blocked cell.
        if (IsBlockedCellReal(flags, *new_pos, ibatch)) {
          // If the source of the ray starts in a blocked cell, we'll never exit
          // the while loop below, also our loop invariant is that new_pos is
          // NEVER allowed to enter a geometry cell. So failing this test means
          // our logic is broken.
          AT_ERROR("Error: Ray source is already in a blocked cell!");
        }
        const uint32_t max_count = 4;  // TODO(tompson): high enough?
        // Note: we need to spin here because while we backoff a blocked cell that
        // is a unit step away, there might be ANOTHER blocked cell along the ray
        // which is less than a unit step away.
        for (uint32_t count = 0; count <= max_count; count++) {
          if (!IsBlockedCellReal(flags, next_pos, ibatch)) {
            break;
          }
          if (count == max_count) {
            AT_ERROR("Error: Cannot find non-geometry point (infinite loop)!");
          }
  
          // Calculate the center of the blocker cell.
          T next_pos_ctr = at::infer_type(pos).zeros({3});
          T ix = getType(pos.type().backend(),at::kInt).zeros({3});
          GetPixelCenter(next_pos, ix);
          next_pos_ctr = ix.toType(at::infer_type(pos)) + 0.5;
  
          if (!IsBlockedCellReal(flags, next_pos_ctr, ibatch)) {
            // Sanity check. This is redundant because IsBlockedCellReal USES
            // GetPixelCenter to sample the FlagGrid. But keep this here
            // just in case the implementation changes.
            AT_ERROR("Error: Center of blocker cell is not a blocker!");
          }
          if (IsOutOfDomainReal(next_pos_ctr, flags)) {
            AT_ERROR("Error: Center of blocker cell is out of the domain!");
          }
          T ipos = at::infer_type(pos).zeros({3});
          const bool hit = calcRayBoxIntersection(*new_pos, dt, next_pos_ctr,
                                                  hit_margin, &ipos);
          if (!hit) {
            // This can happen in very rare cases if the ray box
            // intersection test fails because of floating point round off.
            // It can also happen if the simulation becomes unstable (maybe with a
            // poorly trained model) and the velocity values are extremely high.
            
            // In this case, fall back to simply returning new_pos (for which the
            // loop invariant guarantees is a valid point).
            return true;
          }
   
          next_pos = ipos;
  
          // There's a nasty corner case here. It's when the cell we were trying
          // to step to WAS a blocker, but the ray passed through a blocker to get
          // there (i.e. our step size didn't catch the first blocker). If this is
          // the case we need to do another intersection test, but this time with
          // the ray point destination that is the closer cell.
          // --> There's nothing to do. The outer while loop will try another
          // intersection for us.
        }
        
        // At this point next_pos is guaranteed to be within the domain and
        // not within a solid cell.
        (*new_pos) = next_pos;
  
        // Do some sanity checks.
        if (IsBlockedCellReal(flags, *new_pos, ibatch)) {
          AT_ERROR("Error: case 2 entered geometry!");
        }
        if (IsOutOfDomainReal(*new_pos, flags)) {
          AT_ERROR("Error: case 2 exited bounds!");
        }
        return true;
      }
  
      // Otherwise, update the position to the current step location.
      (*new_pos) = next_pos;
  
      // Do some sanity checks and check the loop invariant.
      if (IsBlockedCellReal(flags, *new_pos, ibatch)) {
        AT_ERROR("Error: correctness assertion broken. Loop entered geometry!");
      }
      if (IsOutOfDomainReal(*new_pos, flags)) {
        AT_ERROR("Error: correctness assertion broken. Loop exited bounds!");
      }
  
      cur_length += cur_step;
    }
  
    // Finally, yet another set of checks, just in case.
    if (IsOutOfDomainReal(*new_pos, flags)) {
      AT_ERROR("Error: CalcLineTrace returned an out of domain cell!");
    }
    if (IsBlockedCellReal(flags, *new_pos, ibatch)) {
      AT_ERROR("Error: CalcLineTrace returned a blocked cell!");
    }
  
    return false;
  }

} // namespace fluid 
