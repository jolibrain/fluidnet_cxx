#include "solveLinearSys.h"

namespace fluid {

void solveLinearSystemJacobi
(
   T* tensor_p,
   T* tensor_flags,
   T* tensor_div,
   T* tensor_p_prev,
   T* tensor_p_delta,
   T* tensor_p_delta_norm,
   const bool is_3d,
   const float p_tol,
   const int max_iter,
   const bool verbose
) {
  // TODO: check sizes as was done in Lua stack 
  if (max_iter < 1) {
     AT_ERROR("At least 1 iteration is needed (maxIter < 1)");
  }

  FlagGrid flags(*tensor_flags, is_3d);
  RealGrid pressure(*tensor_p, is_3d);
  RealGrid pressure_prev(*tensor_p_prev, is_3d);
  RealGrid div(*tensor_div, is_3d);
  
  // Initialize the pressure to zero.
  tensor_p->zero_();
  tensor_p_prev->zero_();
  
  // Start with the output of the next iteration going to pressure.
  RealGrid* cur_pressure = &pressure;
  RealGrid* cur_pressure_prev = &pressure_prev;
  
  const int32_t nbatch = flags.nbatch();
  const int64_t xsize = flags.xsize();
  const int64_t ysize = flags.ysize();
  const int64_t zsize = flags.zsize();
  const int64_t numel = xsize * ysize * zsize;
 
  T residual;
  T at_zero = infer_type(*tensor_p).scalarTensor(0);
  int64_t iter = 0;
  while (true) {
    const int32_t bnd =1;
    int32_t b, k, j, i;
    // Kernel: Jacobi Iteration
    for (b = 0; b < nbatch; b++) {
      for (k = 0; k < zsize; k++) {
        for (j = 0; j < ysize; j++) {
          for (i = 0; i < xsize; i++) {
            
             if (i < bnd || i > flags.xsize() - 1 - bnd ||
                 j < bnd || j > flags.ysize() - 1 - bnd ||
                 (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
               (*cur_pressure)(i, j, k, b) = 0;  // Zero pressure on the border.
             continue;
             }

             if (flags.isObstacle(i, j, k, b)) {
               (*cur_pressure)(i, j, k, b) = 0;
             continue;
             }
           
             // Otherwise in a fluid or empty cell.
             // TODO(tompson): Is the logic here correct? Should empty cells be non-zero?
             const T divergence = div(i, j, k, b);
           
             // Get all the neighbors
             const T pC = (*cur_pressure_prev)(i, j, k, b);
           
             T p1 = (*cur_pressure_prev)(i - 1, j, k, b);
             T p2 = (*cur_pressure_prev)(i + 1, j, k, b);
             T p3 = (*cur_pressure_prev)(i, j - 1, k, b);
             T p4 = (*cur_pressure_prev)(i, j + 1, k, b);
             T p5 = flags.is_3d() ? (*cur_pressure_prev)(i, j, k - 1, b) : at_zero;
             T p6 = flags.is_3d() ? (*cur_pressure_prev)(i, j, k + 1, b) : at_zero;
             if (flags.isObstacle(i - 1, j, k, b)) {
               p1 = pC;
             }
             if (flags.isObstacle(i + 1, j, k, b)) {
               p2 = pC;
             }
             if (flags.isObstacle(i, j - 1, k, b)) {
               p3 = pC;
             }
             if (flags.isObstacle(i, j + 1, k, b)) {
               p4 = pC;
             }
             if (flags.is_3d() && flags.isObstacle(i, j, k - 1, b)) {
               p5 = pC;
             }
             if (flags.is_3d() && flags.isObstacle(i, j, k + 1, b)) {
               p6 = pC;
             }
           
             const T denom = flags.is_3d() ? infer_type(*tensor_p).scalarTensor(6) : infer_type(*tensor_p).scalarTensor(4);
             const T v = (p1 + p2 + p3 + p4 + p5 + p6 + divergence) / denom;
             (*cur_pressure)(i, j, k, b) = v;

          } 
        } 
      }
    }
    // Currrent iteration output is now in cur_pressure
    
    // Now calculate the change in pressure up to a sign (the sign might be 
    // incorrect, but we don't care).
    // p_delta = p - p_prev
    at::sub_out(*tensor_p_delta, *tensor_p, *tensor_p_prev);
    (*tensor_p_delta).resize_({nbatch, numel}); 
    // Calculate L2 norm over dim 2.
    at::norm_out(*tensor_p_delta_norm, *tensor_p_delta, at::Scalar(2), 1);
    (*tensor_p_delta).resize_({nbatch, 1, zsize, ysize, xsize});
    residual = tensor_p_delta_norm->max();
    if (verbose) {
      std::cout << "Jacobi iteration " << (iter + 1) << ":residual "
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
    RealGrid* tmp = cur_pressure;
    cur_pressure = cur_pressure_prev;
    cur_pressure_prev = tmp;
  } // end while

  // If we terminated with the cur_pressure pointing to the tmp array, then we
  // have to copy the pressure back into the output tensor.
  if (cur_pressure == &pressure_prev) {
    tensor_p->copy_(*tensor_p_prev);  // p = p_prev
  }

  // TODO: write mean-subtraction (FluidNet does it in Lua)

}

} // namespace fluid 
