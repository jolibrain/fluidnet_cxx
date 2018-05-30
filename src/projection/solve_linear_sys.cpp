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
   T& tensor_p,
   T& tensor_flags,
   T& tensor_div,
   const bool is_3d,
   const float p_tol = 1e-5,
   const int max_iter = 1000,
   const bool verbose = false
) {
  // Check arguments.
  AT_ASSERT(tensor_p.dim() == 5 && tensor_flags.dim() == 5 && tensor_div.dim() == 5,
             "Dimension mismatch");
  AT_ASSERT(tensor_flags.size(1) == 1, "flags is not scalar");
  int bsz = tensor_flags.size(0);
  int d = tensor_flags.size(2);
  int h = tensor_flags.size(3);
  int w = tensor_flags.size(4);
  AT_ASSERT(tensor_p.is_same_size(tensor_flags), "size mismatch");
  AT_ASSERT(tensor_div.is_same_size(tensor_flags), "size mismatch");
  if (!is_3d) {
    AT_ASSERT(d == 1, "d > 1 for a 2D domain");
  } 

  AT_ASSERT(tensor_p.is_contiguous() && tensor_flags.is_contiguous() &&
            tensor_div.is_contiguous(), "Input is not contiguous");
   
  T tensor_p_prev = infer_type(tensor_p).zeros({bsz, 1, d, h, w});
  T tensor_p_delta = infer_type(tensor_p).zeros({bsz, 1, d, h, w});
  T tensor_p_delta_norm = infer_type(tensor_p).zeros({bsz});
   
  if (max_iter < 1) {
     AT_ERROR("At least 1 iteration is needed (maxIter < 1)");
  }

  FlagGrid flags(tensor_flags, is_3d);
  RealGrid pressure(tensor_p, is_3d);
  RealGrid pressure_prev(tensor_p_prev, is_3d);
  RealGrid div(tensor_div, is_3d);
  
  // Initialize the pressure to zero.
  tensor_p.zero_();
  
  // Start with the output of the next iteration going to pressure.

  auto cur_pressure = tensor_p.accessor<float,5>();
  auto cur_pressure_prev = tensor_p_prev.accessor<float,5>();
  auto div_a = tensor_div.accessor<float,5>();
  //RealGrid* cur_pressure_prev = &pressure_prev;
  
  const int32_t nbatch = flags.nbatch();
  const int64_t xsize = flags.xsize();
  const int64_t ysize = flags.ysize();
  const int64_t zsize = flags.zsize();
  const int64_t numel = xsize * ysize * zsize;

  T residual;
  T at_zero = infer_type(tensor_p).scalarTensor(0);

  int64_t iter = 0;
  while (true) {
    const int32_t bnd =1;
    int32_t b, k, j, i;
    // Kernel: Jacobi Iteration
    for (b = 0; b < nbatch; b++) {
#pragma omp parallel for collapse(3) private (k, j, i)
      for (k = 0; k < zsize; k++) {
        for (j = 0; j < ysize; j++) {
          for (i = 0; i < xsize; i++) {
             if (i < bnd || i > flags.xsize() - 1 - bnd ||
                 j < bnd || j > flags.ysize() - 1 - bnd ||
                (flags.is_3d() && (k < bnd || k > flags.zsize() - 1 - bnd))) {
               cur_pressure[b][0][k][j][i] = 0;  // Zero pressure on the border.
               continue;
             }

             if (flags.isObstacle(i, j, k, b)) {
               cur_pressure[b][0][k][j][i] = 0;
               continue;
             }
           
             // Otherwise in a fluid or empty cell.
             // TODO(tompson): Is the logic here correct? Should empty cells be non-zero?
#pragma omp atomic
             const float divergence = div_a[b][0][k][j][i];
           
             // Get all the neighbors
#pragma omp atomic
             const float pC = cur_pressure_prev[b][0][k][j][i];
#pragma omp atomic           
             float p1 = cur_pressure_prev[b][0][k][j][i-1];
#pragma omp atomic
             float p2 = cur_pressure_prev[b][0][k][j][i+1];
#pragma omp atomic
             float p3 = cur_pressure_prev[b][0][k][j-1][i];
#pragma omp atomic
             float p4 = cur_pressure_prev[b][0][k][j+1][i];
#pragma omp atomic
             float p5 = flags.is_3d() ? cur_pressure_prev[b][0][k-1][j][i] : 0;
#pragma omp atomic
             float p6 = flags.is_3d() ? cur_pressure_prev[b][0][k+1][j][i] : 0;
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
           
             const float denom = flags.is_3d() ? 6 : 4;
#pragma omp atomic
             const float v = (p1 + p2 + p3 + p4 + p5 + p6 + divergence) / denom;
#pragma omp atomic
             cur_pressure[b][0][k][j][i] = v;

          } 
        } 
      }
    }
    // Currrent iteration output is now in cur_pressure
    
    // Now calculate the change in pressure up to a sign (the sign might be 
    // incorrect, but we don't care).
    // p_delta = p - p_prev
    at::sub_out(tensor_p_delta, tensor_p, tensor_p_prev);
    tensor_p_delta.resize_({nbatch, numel}); 
    // Calculate L2 norm over dim 2.
    at::norm_out(tensor_p_delta_norm, tensor_p_delta, at::Scalar(2), 1);
    tensor_p_delta.resize_({nbatch, 1, zsize, ysize, xsize});
    residual = tensor_p_delta_norm.max();
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
    auto tmp = cur_pressure;
    cur_pressure = cur_pressure_prev;
    cur_pressure_prev = tmp;
  } // end while

  // If we terminated with the cur_pressure pointing to the tmp array, then we
  // have to copy the pressure back into the output tensor.
  //if (cur_pressure == &pressure_prev) {
  //  tensor_p.copy_(tensor_p_prev);  // p = p_prev
  //}

  // TODO: write mean-subtraction (FluidNet does it in Lua)
  return at::Scalar(residual).toFloat();
}

} // namespace fluid 
