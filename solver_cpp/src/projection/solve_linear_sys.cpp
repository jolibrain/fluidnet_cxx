#include <memory>
#include "solve_linear_sys.h"

namespace fluid {

// *****************************************************************************
// solveLinearSystemJacobi
// *****************************************************************************

// Solve the linear system using the Jacobi method.
// Note: Since we don't receive a velocity field, we need to receive the is3D
// flag from the caller.
// 
// input p: The output pressure field (i.e. the solution to A * p = div)
// input flags: The input flag grid.
// input div: The velocity divergence.
// input is3D: If true then we expect a 3D domain.
// input pTol: OPTIONAL (default = 1e-5), ||p - p_prev|| termination cond.
// input maxIter: OPTIONAL (default = 1000), max number of Jacobi iterations.
// input verbose: OPTIONAL (default = false), if true print out iteration res.
// 
// output: the max pTol across the batches.

float solveLinearSystemJacobi
(
   T& p,
   T& flags,
   T& div,
   const bool is3D,
   const float p_tol = 1e-5,
   const int max_iter = 1000,
   const bool verbose = false
) {
  // Check arguments.
  AT_ASSERT(p.dim() == 5 && flags.dim() == 5 && div.dim() == 5,
             "Dimension mismatch");
  AT_ASSERT(flags.size(1) == 1, "flags is not scalar");
  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);
  int numel = d * h * w;
  AT_ASSERT(p.is_same_size(flags), "size mismatch");
  AT_ASSERT(div.is_same_size(flags), "size mismatch");
  if (!is3D) {
    AT_ASSERT(d == 1, "d > 1 for a 2D domain");
  } 

  AT_ASSERT(p.is_contiguous() && flags.is_contiguous() &&
            div.is_contiguous(), "Input is not contiguous");
   
  T p_prev = infer_type(p).zeros({bsz, 1, d, h, w});
  T p_delta = infer_type(p).zeros({bsz, 1, d, h, w});
  T p_delta_norm = infer_type(p).zeros({bsz});
   
  if (max_iter < 1) {
     AT_ERROR("At least 1 iteration is needed (maxIter < 1)");
  }
  
  // Initialize the pressure to zero.
  p.zero_();
  
  // Start with the output of the next iteration going to pressure.
  T* cur_p = &p;
  T* cur_p_prev = &p_prev;
  //RealGrid* cur_pressure_prev = &pressure_prev;

  T residual;
 // T at_zero = infer_type(tensor_p).scalarTensor(0);

  int64_t iter = 0;
  while (true) {
    const int32_t bnd =1;
    // Kernel: Jacobi Iteration
    T mCont = infer_type(flags).ones({bsz, 1, d, h, w}).toType(at::kByte); // Continue mask

    T idx_x = infer_type(flags).arange(0, w).view({1,w}).expand({bsz, d, h, w}).toType(at::kLong);
    T idx_y = infer_type(idx_x).arange(0, h).view({1,h,1}).expand({bsz, d, h, w});
    T idx_z = zeros_like(idx_x);
    if (is3D) {
       idx_z = infer_type(idx_x).arange(0, d).view({1,d,1,1}).expand({bsz, d, h, w});
    }

    T idx_b = infer_type(flags).arange(0, bsz).view({bsz,1,1,1}).toType(at::kLong);
    idx_b = idx_b.expand({bsz,d,h,w});

    T maskBorder = (idx_x < bnd).__or__
                   (idx_x > w - 1 - bnd).__or__
                   (idx_y < bnd).__or__
                   (idx_y > h - 1 - bnd);
    if (is3D) {
        maskBorder = maskBorder.__or__(idx_z < bnd).__or__
                                      (idx_z > d - 1 - bnd);
    }
    maskBorder.unsqueeze_(1);

    // Zero pressure on the border.
    cur_p->masked_fill_(maskBorder, 0);
    mCont.masked_fill_(maskBorder, 0);

    T maskObstacle = flags.eq(TypeObstacle).__and__(mCont);
    cur_p->masked_fill_(maskObstacle, 0);
    mCont.masked_fill_(maskObstacle, 0);
  
    
 
    T zero_f = at::zeros_like(p); // Floating zero
    T zero_l = at::zeros_like(p).toType(at::kLong); // Long zero (for index)
    T zeroBy = at::zeros_like(p).toType(at::kByte); // Long zero (for index)
    // Otherwise, we are in a fluid or empty cell.
    // First, we get all the neighbors.

    T pC = *cur_p_prev;

    T i_l = zero_l.where( (idx_x <=0), idx_x - 1);
    T p1 = zero_f.
        where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, idx_z, idx_y, i_l})
        .unsqueeze(1));

    T i_r = zero_l.where( (idx_x > w - 1 - bnd), idx_x + 1);
    T p2 = zero_f.
        where(idx_x >= (w - 3 - bnd), (*cur_p_prev).index({idx_b, zero_l, idx_z, idx_y, i_r})
        .unsqueeze(1));

    T j_l = zero_l.where( (idx_y <= 0), idx_y - 1);
    T p3 = zero_f.
        where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, idx_z, j_l, idx_x})
        .unsqueeze(1));
    T j_r = zero_l.where( (idx_y > h - 1 - bnd), idx_y + 1);
    T p4 = zero_f.
        where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, idx_z, j_r, idx_x})
        .unsqueeze(1));

    T k_l = zero_l.where( (idx_z <= 0), idx_z - 1);
    T p5 = is3D ? zero_f.
        where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, k_l, idx_y, idx_x})
        .unsqueeze(1)) : zero_f;
    T k_r = zero_l.where( (idx_z > d - 1 - bnd), idx_z + 1);
    T p6 = is3D ? zero_f.
        where(mCont.ne(1), (*cur_p_prev).index({idx_b, zero_l, k_r, idx_y, idx_x})
        .unsqueeze(1)) : zero_f;

    T neighborLeftObs = mCont.__and__(zeroBy.
         where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, idx_y, i_l}).eq(TypeObstacle)).unsqueeze(1)); 
    T neighborRightObs = mCont.__and__(zeroBy.
         where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, idx_y, i_r}).eq(TypeObstacle)).unsqueeze(1)); 
    T neighborBotObs = mCont.__and__(zeroBy.
         where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, j_l, idx_x}).eq(TypeObstacle)).unsqueeze(1)); 
    T neighborUpObs = mCont.__and__(zeroBy.
         where(mCont.ne(1), flags.index({idx_b, zero_l, idx_z, j_r, idx_x}).eq(TypeObstacle)).unsqueeze(1)); 
    T neighborBackObs = zeroBy;
    T neighborFrontObs = zeroBy;

    if (is3D) {
      T neighborBackObs = mCont.__and__(zeroBy.
           where(mCont.ne(1), flags.index({idx_b, zero_l, k_l, idx_y, idx_x}).eq(TypeObstacle)).unsqueeze(1)); 
      T neighborFrontObs = mCont.__and__(zeroBy.
           where(mCont.ne(1), flags.index({idx_b, zero_l, k_r, idx_y, idx_x}).eq(TypeObstacle)).unsqueeze(1)); 
    }
   
    p1.masked_scatter_(neighborLeftObs, pC.masked_select(neighborLeftObs));
    p2.masked_scatter_(neighborRightObs, pC.masked_select(neighborRightObs));
    p3.masked_scatter_(neighborBotObs, pC.masked_select(neighborBotObs));
    p4.masked_scatter_(neighborUpObs, pC.masked_select(neighborUpObs));
    p5.masked_scatter_(neighborBackObs, pC.masked_select(neighborBackObs));
    p6.masked_scatter_(neighborFrontObs, pC.masked_select(neighborFrontObs));

    const float denom = is3D ? 6 : 4;
    (*cur_p).masked_scatter_(mCont, ((p1 + p2 + p3 + p4 + p5 + p6 + div) / denom)
                                                             .masked_select(mCont));

    // Currrent iteration output is now in cur_pressure
    
    // Now calculate the change in pressure up to a sign (the sign might be 
    // incorrect, but we don't care).
    // p_delta = p - p_prev
    at::sub_out(p_delta, p, p_prev);
    p_delta.resize_({bsz, numel}); 
    // Calculate L2 norm over dim 2.
    at::norm_out(p_delta_norm, p_delta, at::Scalar(2), 1);
    p_delta.resize_({bsz, 1, d, h, w});
    residual = p_delta_norm.max();
    if (verbose) {
      std::cout << "Jacobi iteration " << (iter + 1) << ": residual "
                << residual << std::endl;
    }

    if (at::Scalar(residual).toFloat() < p_tol) {
      if (verbose) {
        std::cout << "Jacobi max residual fell below p_tol (" << p_tol
                  << ") (terminating)" << std::endl;
      }
      break;
    }
 
    iter++;
    if (iter >= max_iter) {
        if (verbose) {
          std::cout << "Jacobi max iteration count (" << max_iter
                    << ") reached (terminating)" << std::endl;
        }
        break; 
    }
 
    // We haven't yet terminated.
    auto tmp = cur_p;
    cur_p = cur_p_prev;
    cur_p_prev = tmp;
  } // end while

  // If we terminated with the cur_pressure pointing to the tmp array, then we
  // have to copy the pressure back into the output tensor.
  if (cur_p == &p_prev) {
    p.copy_(p_prev);  // p = p_prev
  }

  // TODO: write mean-subtraction (FluidNet does it in Lua)
  return at::Scalar(residual).toFloat();
}

} // namespace fluid 
