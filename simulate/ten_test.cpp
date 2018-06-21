#include <vector>
#include <sstream>

#include "ATen/ATen.h"

#include "calc_line_trace.h"
#include "grid/grid_new.h"
#include "grid/cell_type.h"
#include "advection/advect_type.h"
#include "../test/load_manta_data.h"
#include "../test/plot_utils.h"

namespace fluid {

typedef at::Tensor T;

void setWallBcsForward
(
  T& U, T& flags
) {
  // Check arguments.
  AT_ASSERT(U.dim() == 5 && flags.dim() == 5, "Dimension mismatch");
  AT_ASSERT(flags.size(1) == 1, "flags is not scalar");
  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);

  bool is3D = (U.size(1) == 3);
  if (!is3D) {
     AT_ASSERT(d == 1, "2D velocity field but zdepth > 1");
     AT_ASSERT(U.size(1) == 2, "2D velocity field must have only 2 channels");
  }
  AT_ASSERT((U.size(0) == bsz && U.size(2) == d &&
             U.size(3) == h && U.size(4) == w), "Size mismatch");

  AT_ASSERT(U.is_contiguous() && flags.is_contiguous(),
            "Input is not contiguous");

  T i = infer_type(flags).arange(0, w).view({1,w}).expand({bsz, d, h, w}).toType(at::kLong);
  T j = infer_type(i).arange(0, h).view({1,h,1}).expand({bsz, d, h, w});
  T k = zeros_like(i);
  if (is3D) {
     k = infer_type(i).arange(0, d).view({1,d,1,1}).expand({bsz, d, h, w});
  }
  T zero = zeros_like(i);
  T zeroBy = zero.toType(at::kByte);

  T idx_b = infer_type(i).arange(0, bsz).view({bsz,1,1,1});
  idx_b = idx_b.expand({bsz,d,h,w});

  T mCont = ones_like(zeroBy);

  T cur_fluid = flags.eq(TypeFluid).squeeze(1);
  T cur_obs = flags.eq(TypeObstacle).squeeze(1);
  T mNotFluidNotObs = cur_fluid.ne(1).__and__(cur_obs.ne(1));
  mCont.masked_fill_(mNotFluidNotObs, 0);

  T obst100 = zeroBy.where( i <= 0, (flags.index({idx_b, zero, k, j, i-1}).eq(TypeObstacle))).__and__(mCont);
  U.select(1,0).masked_fill_(obst100, 0);

  T obs_fluid100 = zeroBy.where( i <= 0, (flags.index({idx_b, zero, k, j, i-1}).eq(TypeFluid))).
   __and__(cur_obs).__and__(mCont);
  U.select(1,0).masked_fill_(obs_fluid100, 0);

  T obst010 = zeroBy.where( j <= 0, (flags.index({idx_b, zero, k, j-1, i}).eq(TypeObstacle))).__and__(mCont);
  U.select(1,1).masked_fill_(obst010, 0);

  T obs_fluid010 = zeroBy.where( j <= 0, (flags.index({idx_b, zero, k, j-1, i}).eq(TypeFluid))).
   __and__(cur_obs).__and__(mCont);
  U.select(1,1).masked_fill_(obs_fluid010, 0);
 
  if (is3D) {
    T obst001 = zeroBy.where( k <= 0, (flags.index({idx_b, zero, k-1, j, i}).eq(TypeObstacle))).__and__(mCont);
    U.select(1,2).masked_fill_(obst001, 0);

    T obs_fluid001 = zeroBy.where( k <= 0, (flags.index({idx_b, zero, k-1, j, i}).eq(TypeFluid))).
   __and__(cur_obs).__and__(mCont);
    U.select(1,2).masked_fill_(obs_fluid001, 0);
  }

// TODO: implement TypeStick BCs
}

void addBuoyancy
(
  T& U, T& flags, T& density, T& gravity,
  const float dt
) {
  // Argument check
  AT_ASSERT(U.dim() == 5 && flags.dim() == 5 && density.dim() == 5,
    "Dimension mismatch");
  AT_ASSERT(flags.size(1) == 1, "flags is not scalar");
  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);

  bool is3D = (U.size(1) == 3);

  int bnd = 1;

  if (!is3D) {
     AT_ASSERT(d == 1, "2D velocity field but zdepth > 1");
     AT_ASSERT(U.size(1) == 2, "2D velocity field must have only 2 channels");
  }
  AT_ASSERT((U.size(0) == bsz && U.size(2) == d &&
             U.size(3) == h && U.size(4) == w), "Size mismatch");
  AT_ASSERT(density.is_same_size(flags), "Size mismatch");

  AT_ASSERT(U.is_contiguous() && flags.is_contiguous() &&
            density.is_contiguous(), "Input is not contiguous");

  AT_ASSERT(gravity.dim() == 1 && gravity.size(0) == 3,
           "Gravity must be a 3D vector (even in 2D)"); 

  T strength = - gravity * (dt / fluid::ten::getDx(flags));

  T i = infer_type(flags).arange(0, w).view({1,w}).expand({bsz, d, h, w}).toType(at::kLong);
  T j = infer_type(i).arange(0, h).view({1,h,1}).expand({bsz, d, h, w});
  T k = zeros_like(i);
  if (is3D) {
     k = infer_type(i).arange(0, d).view({1,d,1,1}).expand({bsz, d, h, w});
  }
  T zero = zeros_like(i);
  T zeroBy = zero.toType(at::kByte);
  T zero_f = zero.toType(infer_type(density));

  T idx_b = infer_type(i).arange(0, bsz).view({bsz,1,1,1});
  idx_b = idx_b.expand({bsz,d,h,w});

  T maskBorder = (i < bnd).__or__
                 (i > w - 1 - bnd).__or__
                 (j < bnd).__or__
                 (j > h - 1 - bnd);
  if (is3D) {
      maskBorder = maskBorder.__or__(k < bnd).__or__
                                    (k > d - 1 - bnd);
  }
  maskBorder = maskBorder.unsqueeze(1);

  // No buoyancy on the border. Set continue (mCont) to false.
  T mCont = ones_like(zeroBy).unsqueeze(1);
  mCont.masked_fill_(maskBorder, 0);

  T isFluid = flags.eq(TypeFluid).__and__(mCont);
  mCont.masked_fill_(isFluid.ne(1), 0);
  mCont.squeeze_(1);

  T fluid100 = zeroBy.where( i <= 0, (flags.index({idx_b, zero, k, j, i-1}).eq(TypeFluid))).__and__(mCont);
  T factor = 0.5 * strength[0] * (density.squeeze(1) +
             zero_f.where(i <= 0, density.index({idx_b, zero, k, j, i-1})) );
  U.select(1,0).masked_scatter_(fluid100, (U.select(1,0) + factor).masked_select(fluid100));

  T fluid010 = zeroBy.where( j <= 0, (flags.index({idx_b, zero, k, j-1, i}).eq(TypeFluid))).__and__(mCont);
  factor = 0.5 * strength[1] * (density.squeeze(1) +
             zero_f.where( j <= 0, density.index({idx_b, zero, k, j-1, i})) );
  U.select(1,1).masked_scatter_(fluid010, (U.select(1,1) + factor).masked_select(fluid010));

  if (is3D) {
    T fluid001 = zeroBy.where( j <= 0, (flags.index({idx_b, zero, k-1, j, i}).eq(TypeFluid))).__and__(mCont);
    factor = 0.5 * strength[2] * (density.squeeze(1) +
               zero_f.where(k <= 1, density.index({idx_b, zero, k-1, j, i})) );
    U.select(1,2).masked_scatter_(fluid001, (U.select(1,2) + factor).masked_select(fluid001));

  }

}

void addGravity
(
  T& U, T& flags, T& gravity,
  const float dt
) {
  // Argument check
  AT_ASSERT(U.dim() == 5 && flags.dim() == 5, "Dimension mismatch");
  AT_ASSERT(flags.size(1) == 1, "flags is not scalar");
  int bsz = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);

  bool is3D = (U.size(1) == 3);

  int bnd = 1;
  if (!is3D) {
     AT_ASSERT(d == 1, "2D velocity field but zdepth > 1");
     AT_ASSERT(U.size(1) == 2, "2D velocity field must have only 2 channels");
  }
  AT_ASSERT((U.size(0) == bsz && U.size(2) == d &&
             U.size(3) == h && U.size(4) == w), "Size mismatch");

  AT_ASSERT(U.is_contiguous() && flags.is_contiguous(), "Input is not contiguous");

  AT_ASSERT(gravity.dim() == 1 && gravity.size(0) == 3,
           "Gravity must be a 3D vector (even in 2D)");
 
  T force = gravity * (dt / fluid::ten::getDx(flags));

  T i = infer_type(flags).arange(0, w).view({1,w}).expand({bsz, d, h, w}).toType(at::kLong);
  T j = infer_type(i).arange(0, h).view({1,h,1}).expand({bsz, d, h, w});
  T k = zeros_like(i);
  if (is3D) {
     k = infer_type(i).arange(0, d).view({1,d,1,1}).expand({bsz, d, h, w});
  }
  T zero = zeros_like(i);
  T zeroBy = zero.toType(at::kByte);
  T zero_f = zero.toType(infer_type(U));

  T idx_b = infer_type(i).arange(0, bsz).view({bsz,1,1,1});
  idx_b = idx_b.expand({bsz,d,h,w});

  T maskBorder = (i < bnd).__or__
                 (i > w - 1 - bnd).__or__
                 (j < bnd).__or__
                 (j > h - 1 - bnd);
  if (is3D) {
      maskBorder = maskBorder.__or__(k < bnd).__or__
                                    (k > d - 1 - bnd);
  }
  maskBorder = maskBorder.unsqueeze(1);

  // No buoyancy on the border. Set continue (mCont) to false.
  T mCont = ones_like(zeroBy).unsqueeze(1);
  mCont.masked_fill_(maskBorder, 0);

  T cur_fluid = flags.eq(TypeFluid).__and__(mCont);
  T cur_empty = flags.eq(TypeEmpty).__and__(mCont);

  T mNotFluidNotEmpt = cur_fluid.ne(1).__and__(cur_empty.ne(1));
  mCont.masked_fill_(mNotFluidNotEmpt, 0);

  mCont.squeeze_(1);

  T fluid100 = (zeroBy.where( i <= 0, (flags.index({idx_b, zero, k, j, i-1}).eq(TypeFluid)))
  .__or__(( zeroBy.where( i <= 0, (flags.index({idx_b, zero, k, j, i-1}).eq(TypeEmpty))))
  .__and__(cur_fluid.squeeze(1)))).__and__(mCont);
  U.select(1,0).masked_scatter_(fluid100, (U.select(1,0) + force[0]).masked_select(fluid100));
    
  T fluid010 = (zeroBy.where( j <= 0, (flags.index({idx_b, zero, k, j-1, i}).eq(TypeFluid)))
  .__or__(( zeroBy.where( j <= 0, (flags.index({idx_b, zero, k, j-1, i}).eq(TypeEmpty))))
  .__and__(cur_fluid.squeeze(1))) ).__and__(mCont);
  U.select(1,1).masked_scatter_(fluid010, (U.select(1,1) + force[1]).masked_select(fluid010));

  if (is3D) {
    T fluid001 = (zeroBy.where( k <= 0, (flags.index({idx_b, zero, k-1, j, i}).eq(TypeFluid)))
    .__or__(( zeroBy.where( k <= 0, (flags.index({idx_b, zero, k-1, j, i}).eq(TypeEmpty))))
    .__and__(cur_fluid.squeeze(1)))).__and__(mCont);
    U.select(1,2).masked_scatter_(fluid001, (U.select(1,2) + force[2]).masked_select(fluid001));
  }

}


} // namespace fluid


void testSetWallBcs(int dim, std::string fnInput, std::string fnOutput) {
   std::string fn = std::to_string(dim) + "d_" + fnInput;
   at::Tensor undef1;
   at::Tensor U;
   at::Tensor flags;
   bool is3D;
   loadMantaBatch(fn, undef1, U, flags, undef1, is3D);

   assertNotAllEqual(U);
   assertNotAllEqual(flags);

   if (!(is3D == (dim == 3))) {
      AT_ERROR("Error with is3D at input");}

   if (!(at::Scalar(at::max(at::abs(flags - fluid::TypeFluid))).toFloat() > 0)) {
      AT_ERROR("All cells are fluid at input.");
   }

   fn = std::to_string(dim) + "d_" + fnOutput;
   at::Tensor undef2;
   at::Tensor UManta;
   at::Tensor flagsManta;
   loadMantaBatch(fn, undef2, UManta, flagsManta, undef2, is3D);

   if (!(is3D == (dim == 3))) {
      AT_ERROR("Error with is3D at output.");}

   if (!(flags.equal(flagsManta))) {
      AT_ERROR("Flags changed!"); }

   at::Tensor UOurs = U.clone();

   fluid::setWallBcsForward(UOurs, flags);
   at::Tensor err = UOurs - UManta;

   float err_abs = at::Scalar(at::max(at::abs(err))).toFloat();
   float precision = 1e-6;

   std::string ss = "Test " + std::to_string(dim) + "d " +
     fnOutput + ": FAILED (max error is " + std::to_string(err_abs) + ").";
   const char *css = ss.c_str();
   AT_ASSERT(err_abs < precision, css);

}

void setWallBcs() {
  for (int dim = 2; dim < 4; dim++){
    testSetWallBcs(dim, "advect.bin", "setWallBcs1.bin");
    testSetWallBcs(dim, "gravity.bin", "setWallBcs2.bin");
    testSetWallBcs(dim, "solvePressure.bin", "setWallBcs3.bin");
  }
   std::cout << "Set Wall Bcs ----------------------- [PASS]" << std::endl;
}

at::Tensor getGravity(int dim, at::Tensor& flags) {
   float gStrength = fluid::ten::getDx(flags) / 4;
   at::Tensor gravity = CPU(at::kFloat).arange(1,4);
   if (dim == 2) {
      gravity[2] = 0;
   }
   gravity.div_(gravity.norm()).mul_(gStrength);
   return gravity;
}

void addBuoyancy() {
   for (int dim = 2; dim < 4; dim++) {
      // Load the input Manta data
      std::string fn = std::to_string(dim) + "d_setWallBcs1.bin";
      at::Tensor undef1;
      at::Tensor U;
      at::Tensor flags;
      at::Tensor density;
      bool is3D;
      loadMantaBatch(fn, undef1, U, flags, density, is3D);
      assertNotAllEqual(U);
      assertNotAllEqual(flags);
      assertNotAllEqual(density);
      AT_ASSERT(is3D == (dim == 3), "3D boolean is inconsistent");

      // Now load the output Manta data.
      fn = std::to_string(dim) + "d_buoyancy.bin";
      at::Tensor undef2;
      at::Tensor UManta;
      at::Tensor flagsManta;
      loadMantaBatch(fn, undef2, UManta, flagsManta, undef2, is3D);
      assertNotAllEqual(UManta);
      assertNotAllEqual(flagsManta);
      AT_ASSERT(is3D == (dim == 3), "3D boolean is inconsistent");
      AT_ASSERT(flags.equal(flagsManta) , "flags changed!");

      AT_ASSERT(at::Scalar(at::max(at::abs(U - UManta))).toFloat() > 1e-5,
                "No velocities changed in Manta!");

      // Use own addBuoyancy
      at::Tensor UOurs = U.clone();
      at::Tensor gravity = getGravity(dim, flags);
      float dt = 0.1;
      fluid::addBuoyancy(UOurs, flags, density, gravity, dt);
      at::Tensor err = UManta - UOurs;
      float err_abs = at::Scalar(at::max(at::abs(err))).toFloat();
      float precision = 1e-6;

      std::string ss = "Test " + std::to_string(dim) +
         "d addBuoyancy: FAILED (max error is " + std::to_string(err_abs)
         + ").";
      const char *css = ss.c_str();
      AT_ASSERT(err_abs < precision, css);
   }
   std::cout << "Add Buoyancy:------------------[PASS]" << std::endl;
}

void addGravity() {
   for (int dim = 2; dim < 4; dim++) {
      // Load the input Manta data
      std::string fn = std::to_string(dim) + "d_vorticityConfinement.bin";
      at::Tensor undef1;
      at::Tensor U;
      at::Tensor flags;
      bool is3D;
      loadMantaBatch(fn, undef1, U, flags, undef1, is3D);
      assertNotAllEqual(U);
      assertNotAllEqual(flags);
      AT_ASSERT(is3D == (dim == 3), "3D boolean is inconsistent");

      // Now load the output Manta data.
      fn = std::to_string(dim) + "d_gravity.bin";
      at::Tensor undef2;
      at::Tensor UManta;
      at::Tensor flagsManta;
      loadMantaBatch(fn, undef2, UManta, flagsManta, undef2, is3D);
      assertNotAllEqual(UManta);
      assertNotAllEqual(flagsManta);
      AT_ASSERT(is3D == (dim == 3), "3D boolean is inconsistent");
      AT_ASSERT(flags.equal(flagsManta) , "flags changed!");

      AT_ASSERT(at::Scalar(at::max(at::abs(U - UManta))).toFloat() > 1e-5,
                "No velocities changed in Manta!");

      // Use own addBuoyancy
      at::Tensor UOurs = U.clone();
      at::Tensor gravity = getGravity(dim, flags);
      float dt = 0.1;
      fluid::addGravity(UOurs, flags, gravity, dt);
      at::Tensor err = UManta - UOurs;
      float err_abs = at::Scalar(at::max(at::abs(err))).toFloat();
      float precision = 1e-6;

      std::string ss = "Test " + std::to_string(dim) +
         "d addGravity: FAILED (max error is " + std::to_string(err_abs)
         + ").";
      const char *css = ss.c_str();
      AT_ASSERT(err_abs < precision, css);
   }
   std::cout << "Add Gravity:------------------[PASS]" << std::endl;
}

int main() {

  auto && Tfloat = CPU(at::kFloat);

//  int dim = 2;
// 
//      std::string fn = std::to_string(dim) + "d_gravity.bin";
//      T undef1;
//      T U;
//      T flags;
//      T density;
//      bool is3D ;
//      loadMantaBatch(fn, undef1, U, flags, density, is3D);
//      assertNotAllEqual(U);
//      assertNotAllEqual(flags);
//
//      AT_ASSERT(is3D == (dim == 3), "Failed assert is3D");
//      fn = std::to_string(dim) + "d_correctVelocity.bin";
//      T undef2;
//      T pressure;
//      T UManta;
//      T flagsManta;
//      loadMantaBatch(fn, pressure, UManta, flagsManta, undef2, is3D);
//      AT_ASSERT(is3D == (dim == 3), "Failed assert is3D");
//
//      AT_ASSERT(flags.equal(flagsManta), "Flags changed!");
//      AT_ASSERT(at::Scalar(at::max(at::abs(U - UManta))).toFloat() > 1e-5, "No velocities changed in Manta!");

//  int b = flags.size(0);
//  int d = flags.size(2);
//  int h = flags.size(3);
//  int w = flags.size(4);
//
setWallBcs();
addBuoyancy();
addGravity();

  int b = 1;
  int d = 1;
  int h = 4;
  int w = 4;

//  is3D = true;
//  T index_x = CPU(at::kInt).arange(0, h).view({h}).expand({d,h,w});
//  T index_y = CPU(at::kInt).arange(0, w).view({w, 1}).expand({d,h,w});
//  T index_z;
//  if (is3D) {
//     index_z = CPU(at::kInt).arange(0, d).view({d, 1 , 1}).expand({d,h,w});
//  }
//  T index_ten;
//
//  if (!is3D) {
//    index_ten = at::stack({index_x, index_y}, 0).view({1,2,d,h,w});
//  }
//  else { 
//    index_ten = at::stack({index_x, index_y, index_z}, 0).view({1,3,d,h,w});
//  }
//
//  index_ten = index_ten.expand_as(pos);
  //index_ten.expand_as(pos);
  // std::cout << pos << std::endl;
   
//  T self = Tfloat.rand({b,1,d,h,w}) * 255;
//  T flags = CPU(at::kInt).randint(2, {b, 1, d,h,w}) + 1;
//  T pos = CPU(at::kFloat).rand({b,3,d,h,w}) * (h-1);
//  T new_pos;
//  T pos = CPU(at::kFloat).full({b,3,d,h,w}, 0.5);
//  for (int i = 0; i<w; i++) {
//    for (int j= 0; j < h; j++) {
//      for (int k= 0; k< d; k++) {
//         pos[0][0][k][j][i] = i+0.5;
//         pos[0][1][k][j][i] = j+0.5;
//         pos[0][2][k][j][i] = k+0.5;
//      }
//    }
//  } 
//  T delta = CPU(at::kFloat).full({b,3,d,h,w}, 1);
//  delta.select(1,2).fill_(_(0);  


//  T pos = CPU(at::kFloat).full({b,3,d,h,w}, 0.5);
//  for (int i = 0; i<w; i++) {
//    for (int j= 0; j < h; j++) {
//      for (int k= 0; k< d; k++) {
//         pos[0][0][k][j][i] = i+0.5;
//         pos[0][1][k][j][i] = j+0.5;
//         pos[0][2][k][j][i] = k+0.5;
//      }
//    }
//  } 
//
//  T flags = CPU(at::kInt).ones({b, 1, d,h,w});
//  T new_pos;
//
//  T delta = CPU(at::kFloat).zeros({b,3,d,h,w});
//
//  flags.select(0,0)[0][0][0][1] = 2;
//  flags.select(0,0)[0][0][3][0] = 2;
//
//  delta.select(0,0).select(0,0) = -1;
//  delta[0][0][0][2][2] = -1;
//  delta[0][1][0][2][2] = -2;
//  
//  
//  calcLineTrace(pos, delta, flags,
//                new_pos, true);

//  
//  pos[0][0][0][0][0] = -100;
//
//
//  T tempPos = pos;
//  std::cout << pos << std::endl;
  //clampToDomain(pos, flags);
 
}
