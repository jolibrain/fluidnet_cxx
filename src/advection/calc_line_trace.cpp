#include "calc_line_trace.h"

namespace fluid {

typedef at::Tensor T;

const float hit_margin = 1e-5;
const float epsilon = 1e-12;

void getPixelCenter(const T& pos, T& ix) {
  AT_ASSERT(at::isFloatingType(infer_type(pos).scalarType()), "Error: getPixelCenter expects floating type tensor for pos argument");
  AT_ASSERT(pos.size(1) == 3, "Pos must have 3 channels");
  ix = pos.toType(at::kLong);
}

T isOutOfDomain(const T& pos, const T& flags) {
  AT_ASSERT(at::isFloatingType(infer_type(pos).scalarType()), "Error: isOutOfDomain expects floating type tensor for pos argument");
  // Note: no need to make the difference between 2d and 3d as postion are intially
  // moved by 0.5. If pos is 2d, pos.z = 0.5 and the two last conditions will always
  // be false.
  return( ((pos.select(1,0) <= 0).__or__
           (pos.select(1,0) >= flags.size(4)).__or__
           (pos.select(1,1) <= 0).__or__
           (pos.select(1,1) >= flags.size(3)).__or__
           (pos.select(1,2) <= 0).__or__
           (pos.select(1,2) >= flags.size(2))).unsqueeze(1) );       
}

// Returns true if the cell is blocked. In FluidNet, it couldn't be called on 
// cells outside the domain. However here we don't have any option as we are always
// working with the complete pos tensor. Therefore we work with masks and
// if outOfDomain, return false.
T isBlockedCell(const T& pos, const T& flags) {

  AT_ASSERT(at::isFloatingType(infer_type(pos).scalarType()), "Error: isBlockedCell expects floating type tensor for pos argument");

  int bsz = pos.size(0);
  int d = pos.size(2);
  int h = pos.size(3);
  int w = pos.size(4);

  T ix;
  T idx_b = at::infer_type(pos).arange(0, bsz).view({bsz,1,1,1}).toType(at::kLong);
  idx_b = idx_b.expand({bsz,d,h,w});

  T isOut = isOutOfDomain(pos, flags);

  getPixelCenter(pos, ix);
  ix.masked_fill_(isOut, 0); // If the pos lies outside the domain, the index function
                             // will not work. We put all those cells in (0,0,0) and
                             // we operate with mask so that we don't take those into
                             // account,
 
  T ret = at::zeros_like(flags).toType(at::kByte);
  // ret is false by default. Only operate on cells that are INSIDE the domain, 
  // using the complentary of isOut.
  // Set ret to true when cells are obstacles.
  ret.masked_scatter_(isOut.eq(0), 
            (flags.index({idx_b, {}, ix.select(1,2), ix.select(1,1), ix.select(1,0)})
           .squeeze(4).unsqueeze(1).ne(fluid::TypeFluid)).masked_select(isOut.eq(0)));
  return ret;
}

void clampToDomain(T& pos, const T& flags) {

  pos.select(1,0).clamp((hit_margin), (flags.size(4) - hit_margin));
  pos.select(1,1).clamp((hit_margin), (flags.size(3) - hit_margin));
  pos.select(1,2).clamp((hit_margin), (flags.size(2) - hit_margin));
}

T HitBoundingBox(const T& minB, const T& maxB,
                 const T& origin, const T& dir,
                 const T& mask, T& coord){

  T ret = ones_like(mask);
  T inside = ones_like(mask);
  T quadrant = zeros_like(origin);
  T which_Plane = zeros_like(mask).toType(at::kLong);
  T maxT = zeros_like(origin);
  T candidate_plane = zeros_like(origin);
  coord = zeros_like(origin);

  T maskLT_minB = (origin < minB).__and__(mask);
  T maskGT_maxB = (origin > maxB).__and__(mask);
  T mask_insideB = (origin >= minB).__and__(origin <= maxB).__and__(mask);

  quadrant.masked_fill_(maskLT_minB, LEFT);
  candidate_plane.masked_scatter_(maskLT_minB, minB.masked_select(maskLT_minB));  
  inside.masked_fill_(maskLT_minB.select(1,0).__or__
                     (maskLT_minB.select(1,1)).__or__
                     (maskLT_minB.select(1,2)).unsqueeze(1), 0);

  quadrant.masked_fill_(maskGT_maxB, RIGHT);
  candidate_plane.masked_scatter_(maskGT_maxB, maxB.masked_select(maskGT_maxB));  
  inside.masked_fill_(maskGT_maxB.select(1,0).__or__
                     (maskGT_maxB.select(1,1)).__or__
                     (maskGT_maxB.select(1,2)).unsqueeze(1), 0);
 
  quadrant.masked_fill_(mask_insideB, MIDDLE);

  // Ray origin inside bounding box. 
  coord.masked_scatter_(inside.__and__(mask), origin.masked_select(inside.__and__(mask)));

  // Otherwise, calculte T distances to candidate planes.
  T outside = inside.ne(1).__and__(mask);

  T NotMiddleAndDirNotNull = outside.__and__(quadrant.ne(MIDDLE)).__and__(dir.ne(0)).__and__(mask);
  T MiddleOrDirNull = outside.__and__(quadrant.eq(MIDDLE)).__or__(dir.eq(0)).__and__(mask);
   
  maxT.masked_scatter_(NotMiddleAndDirNotNull, 
                      ((candidate_plane - origin) / dir).masked_select(NotMiddleAndDirNotNull));
  maxT.masked_fill_(MiddleOrDirNull, -1); 
  
  // Get the largest of the maxT's for final choice of intersection.
  T whichPlane = maxT.argmax(1,true);

  T finalCandidate = maxT.max_values(1, true);

  // Check final candidate actually inside the box and calculate the coords (if not).
  T finalCandidateInsideBox = (finalCandidate < 0).__and__(outside).__and__(mask);
  ret.masked_fill_(finalCandidateInsideBox, 0);

  const at::Scalar err_tol = 1e-6;
  
  coord.select(1,0) = where(whichPlane.squeeze(1).eq(0), candidate_plane.select(1,0),
                     origin.select(1,0) + finalCandidate.squeeze(1) * dir.select(1,0));	
  coord.select(1,1) = where(whichPlane.squeeze(1).eq(1), candidate_plane.select(1,1),
                     origin.select(1,1) + finalCandidate.squeeze(1) * dir.select(1,1));	
  coord.select(1,2) = where(whichPlane.squeeze(1).eq(2), candidate_plane.select(1,2),
                     origin.select(1,2) + finalCandidate.squeeze(1) * dir.select(1,2));	

  T coordOutsideBox =(((whichPlane.squeeze(1).ne(0).__and__
                       ((coord.select(1,0) < minB.select(1,0) - err_tol).__or__
                        (coord.select(1,0) > maxB.select(1,0) + err_tol)))
                        .__or__
                       (whichPlane.squeeze(1).ne(1).__and__ 
                       ((coord.select(1,1) < minB.select(1,1) - err_tol).__or__
                        (coord.select(1,1) > maxB.select(1,1) + err_tol)))
                        .__or__
                       (whichPlane.squeeze(1).ne(2).__and__ 
                       ((coord.select(1,2) < minB.select(1,2) - err_tol).__or__
                        (coord.select(1,2) > maxB.select(1,2) + err_tol)))
                      ).__and__(mask.squeeze(1))).unsqueeze(1);
 
  ret.masked_fill_(coordOutsideBox, 0);
  return ret;
}

// calcRayBoxIntersection will calculate the intersection point for the ray
// starting at pos, and pointing along dt (which should be unit length).
// The box is size 1 and is centered at ctr.
T calcRayBoxIntersection(const T& pos, const T& dt, const T& ctr,
                         const float hit_margin, const T& mask,  T& ipos) {
   
  AT_ASSERT(hit_margin > 0, "Error: calcRayBoxIntersection hit_margin < 0");
  T box_min = ctr - 0.5 - hit_margin;
  T box_max = ctr + 0.5 + hit_margin;

  return HitBoundingBox(box_min, box_max,  // box
                        pos, dt,           // ray
                        mask, ipos);
}


// calcRayBorderIntersection will calculate the intersection point for the ray
// starting at pos and pointing to next_pos.
//
// IMPORTANT: This function ASSUMES that the ray actually intersects. Nasty
// things will happen if it does not.
// EDIT(tompson, 09/25/16): This is so important that we'll actually double
// check the input coords anyway.

T calcRayBorderIntersection(const T& pos, const T& next_pos,
                            const T& flags, const float hit_margin,
                            const T& mOutDom,
                            T& ipos) {

   AT_ASSERT(hit_margin > 0, "Error: calcRayBorderIntersection hit_margin < 0");
     
   // Here we only operate on source which are INSIDE the boundary
   // and target location OUTSIDE the boundary. Find a way to assert this.
   T maskAssertInside = isOutOfDomain(pos, flags);
   AT_ASSERT(maskAssertInside.masked_select(mOutDom).
                equal(mOutDom.masked_select(mOutDom).eq(0)), "Error: source location is already outside the domain!");
  
   T maskAssertOutside = isOutOfDomain(next_pos, flags);
   AT_ASSERT(maskAssertOutside.masked_select(mOutDom).
                 equal(mOutDom.masked_select(mOutDom)), "Error: target location is already outside the domain!");

   // Calculate the minimum step length to exit each face and then step that
   // far. The line equation is:
   //   P = gamma * (next_pos - pos) + pos.
   // So calculate gamma required to make P < + margin for each dim
   // independently.
   //   P_i = m --> m - pos_i = gamma * (next_pos_i - pos_i)
   //   --> gamma_i = (m - pos_i) / (next_pos_i - pos_i)

   // minimum step only has ONE channel
   T min_step = full_like(flags.squeeze(1).toType(at::kFloat), INFINITY);

   // Left Face
   T maskLF = next_pos.select(1,0) <= hit_margin;
   T dx = next_pos.select(1,0) - pos.select(1,0); //Squeeze dim 1
   T maskGTEps = (at::abs(dx) >= epsilon).__and__(maskLF).__and__(mOutDom.squeeze(1));
   T xstep = (hit_margin - pos.select(1,0)) / dx;
   min_step.
       masked_scatter_(maskGTEps, (at::min(min_step, xstep)).masked_select(maskGTEps));
   // Front Face
   T maskFF = next_pos.select(1,1) <= hit_margin;
   T dy = next_pos.select(1,1) - pos.select(1,1); //Squeeze dim 1
   maskGTEps = (at::abs(dy) >= epsilon).__and__(maskFF).__and__(mOutDom.squeeze(1)); 
   T ystep = (hit_margin - pos.select(1,1)) / dy;
   min_step.
      masked_scatter_(maskGTEps, at::min(min_step, xstep).masked_select(maskGTEps));
   // Bottom Face
   T maskBF = next_pos.select(1,2) <= hit_margin;
   T dz = next_pos.select(1,2) - pos.select(1,2); //Squeeze dim 1
   maskGTEps = (at::abs(dz) >= epsilon).__and__(maskBF).__and__(mOutDom.squeeze(1)); 
   T zstep = (hit_margin - pos.select(1,2)) / dz;
   min_step.
      masked_scatter_(maskGTEps, at::min(min_step, zstep).masked_select(maskGTEps));
      
   // Also calculate the min step to exit a positive face.
   //   P_i = dim - m --> dim - m - pos_i = gamma * (next_pos_i - pos_i)
   //   --> gamma = (dim - m - pos_i) / (next_pos_i - pos_i)
   
   // Right Face
   T maskRF = next_pos.select(1,0) >= (flags.size(4) - hit_margin);
   dx = next_pos.select(1,0) - pos.select(1,0); //Squeeze dim 1
   maskGTEps = (at::abs(dx) >= epsilon).__and__(maskRF).__and__(mOutDom.squeeze(1)); 
   xstep = (flags.size(4) - hit_margin - pos.select(1,0)) / dx;
   min_step.
       masked_scatter_(maskGTEps, at::min(min_step, xstep).masked_select(maskGTEps));
   // Back Face
   T maskBBF = next_pos.select(1,1) >= (flags.size(3) - hit_margin);
   dy = next_pos.select(1,1) - pos.select(1,1); //Squeeze dim 1
   maskGTEps = (at::abs(dy) >= epsilon).__and__(maskBBF).__and__(mOutDom.squeeze(1)); 
   ystep = (flags.size(3) - hit_margin - pos.select(1,1)) / dy;
   min_step.
      masked_scatter_(maskGTEps, at::min(min_step, ystep).masked_select(maskGTEps));
   // Upper Face
   T maskUF = next_pos.select(1,2) >= (flags.size(2) - hit_margin);
   dz = next_pos.select(1,2) - pos.select(1,2); //Squeeze dim 1
   maskGTEps = (at::abs(dz) >= epsilon).__and__(maskUF).__and__(mOutDom.squeeze(1)); 
   zstep = (flags.size(2) - hit_margin - pos.select(1,2)) / dz;
   min_step.
      masked_scatter_(maskGTEps, at::min(min_step, zstep).masked_select(maskGTEps));

   T maskHit = (min_step >= 0).__and__(min_step < INFINITY).unsqueeze(1);
   ipos = at::zeros_like(pos);
   ipos.masked_scatter_(maskHit, 
          (min_step.unsqueeze(1) * (next_pos - pos) + pos).masked_select(maskHit));
  
   return maskHit;
}

void calcLineTrace(const T& pos, const T& delta, const T& flags,
                T& new_pos, const bool do_line_trace){

  T maskRet = at::zeros_like(flags).toType(at::kByte);
  T mCont = at::ones_like(flags).toType(at::kByte);
  if (!do_line_trace) {
    new_pos = pos + delta;
    return;  //return maskRet;
  }

  // If we are ALREADY outside the domain, don't go further (mask continue = false)
  T isOut = isOutOfDomain(pos, flags);
  mCont.masked_fill_(isOut, 0);
  
  // If we are ALREADY in an obstacle segment, don't go further (mask continue = false)
  T isBlocked = isBlockedCell(pos, flags);
  mCont.masked_fill_(isBlocked, 0);
  new_pos = pos.clone();
  const T length = delta.norm(2, 1, true); // L2 norm in dimension 1 keeping dimension.

  // We are not being asked to step anywhere. Set mask continue to false.
  T infToEps = (length <= epsilon).__and__(mCont);
  mCont.masked_fill_(infToEps, 0);
  
  // The rest of cells are the ones having a true mask in mCont.
  // We will perform the ops on those cells. 
  // The rest of the cells are stopped, i.e mCont is false.
 
  // Figure out the step size in x, y and z for our marching.
  // Only for the non-stopped cells (maskStop is false).
  T dt = at::zeros_like(delta);
  dt.masked_scatter_(mCont, (delta/length).masked_select(mCont));
  // Tompson: We start the line search, by stepping a unit length along the
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

  T cur_length = at::zeros_like(length);
  T next_pos = at::zeros_like(pos);
 
  // The while loop ends when all line traces are done, i.e. maskStop is filled 
  // with ones. Some points will encounter obstacles, others may not. 
  // We return a boolean mask where true means that 
  // the point encountered an obstacle, false otherwise.
  while (!mCont.equal(zeros_like(mCont))) {
    T reachedLength = (cur_length >= length - hit_margin).__and__(mCont);
    mCont.masked_fill_(reachedLength, 0);
    if (mCont.equal(zeros_like(mCont))) {
      break;
    }

    // We haven't stepped far enough. So take a step.
    T cur_step = at::min(length - cur_length, at::ones_like(length));
    next_pos.masked_scatter_( mCont, 
                            ( new_pos + (dt * cur_step) ).masked_select(mCont) );
    // Now check if we went too far,
    // There are two possible cases. We've either stepped out of the domain or
    // entered a blocked cell.
    // Case 1. 'next_pos' exits the grid.
    {
      T mOutDom = isOutOfDomain(next_pos, flags).__and__(mCont);
      T ipos;
      T maskHit = calcRayBorderIntersection(pos, next_pos, flags,
                                        hit_margin, mOutDom, ipos);
      // Case: hit==false && mOutDom = false
      // This is an EXTREMELY rare case. It happens because either the ray is
      // almost parallel to the domain boundary, OR floating point round-off
      // causes the intersection test to fail.

      // In this case, fall back to simply clamping next_pos inside the domain
      // boundary. It's not ideal, but better than a hard failure (the reason
      // why it's wrong is that clamping will bring the point off the ray).
      T clampedIpos = ipos.clone();
      T maskNoHit = maskHit.eq(0).__and__(mOutDom);
      clampedIpos.masked_scatter_(maskNoHit, next_pos.masked_select(maskNoHit)); 
      clampToDomain(clampedIpos, flags);
      ipos.masked_scatter_(maskNoHit, clampedIpos.masked_select(maskNoHit));
 
      // Do some sanity checks.
      // The logic above should always put ipos back hit_margin inside the
      // simulation domain.
      AT_ASSERT(isOutOfDomain(ipos, flags).masked_select(mOutDom).
           equal(mOutDom.masked_select(mOutDom).eq(0)), "Error: case 1 exited bounds!");

      isBlocked = isBlockedCell(ipos, flags).__and__(mOutDom);
      T isAgainstBorder = isBlockedCell(ipos, flags).eq(0).__and__(mOutDom);
 
      // We are up against the border and not in a blocked cell.
      // Change continue to false.
      new_pos.masked_scatter_(isAgainstBorder, ipos.masked_select(isAgainstBorder)); 

      mCont.masked_fill_(isAgainstBorder, 0);

      // We hit the border boundary, but we entered a blocked cell.
      // Continue on to case 2. 
      next_pos.masked_scatter_(isBlocked.__and__(mCont), ipos.masked_select(isBlocked.__and__(mCont)));
    }
    // Case 2. next_pos enters a blocked cell.
    {
      T mBlock = isBlockedCell(next_pos, flags).__and__(mCont);
      AT_ASSERT(isBlockedCell(new_pos, flags).masked_select(mBlock).
           equal(mBlock.masked_select(mBlock).eq(0)), "Error: Ray source is already in a blocked cell!");
      
      const uint32_t max_count = 4; 
      // TODO(tompson): high enough?
      // Note: we need to spin here because while we backoff a blocked cell that
      // is a unit step away, there might be ANOTHER blocked cell along the ray
      // which is less than a unit step away.
      T countMask = mBlock.clone();
      for (uint32_t count = 0; count <= max_count; count++) {
        T mBlockCount = isBlockedCell(next_pos, flags).ne(1).__and__(countMask);

        countMask.masked_fill_(mBlockCount, 0);

        if (countMask.masked_select(mBlock).equal(
                 zeros_like(countMask.masked_select(mBlock)))) {
          break;
        }
       AT_ASSERT(count < max_count, "Error: Cannot find non-geometry point (infinite loop)!");
       
       // Calculate the center of the blocker cell.
       T next_pos_ctr;
       T idx;
       getPixelCenter(next_pos, idx);

       next_pos_ctr = idx.toType(infer_type(pos)) + 0.5;
       AT_ASSERT(isOutOfDomain(next_pos_ctr, flags).eq(0).masked_select(countMask).
           equal(countMask.masked_select(countMask)), "Error: Center of blocker cell is out of the domain!");
     
       T ipos;
       T hit = calcRayBoxIntersection(new_pos, dt, next_pos_ctr,
                                      hit_margin, countMask, ipos);
       mCont.masked_fill_(hit.eq(0).__and__(countMask), 0);
       countMask.masked_fill_(hit.eq(0).__and__(countMask), 0);

       next_pos.masked_scatter_(hit.__and__(countMask),
                               ipos.masked_select(hit.__and__(countMask)));
    } 

    // At this point next_pos is guaranteed to be within the domain and
    // not within a solid cell.
    new_pos.masked_scatter_(mBlock.__and__(mCont),
                            next_pos.masked_select(mBlock.__and__(mCont)));

    // Change continue to false.
    mCont.masked_fill_(mBlock.__and__(mCont), 0);

    } 
 
  // Otherwise, update the position to the current step location
  new_pos.masked_scatter_(mCont, next_pos.masked_select(mCont));
  cur_length.masked_scatter_(mCont, (cur_length+cur_step).masked_select(mCont));
  
  //Check if cur_length < length - hit_margin. Otherwise, set continue to false.
  reachedLength = (cur_length >= length - hit_margin).__and__(mCont);
  mCont.masked_fill_(reachedLength, 0);
    
  }
  return;
}

} // namespace fluid
