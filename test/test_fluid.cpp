#include <sstream>
#include <omp.h>

#include "ATen/ATen.h"

#include "load_manta_data.h"
#include "plot_utils.h"
#include "fluid.h"
#include "init.cpp"

// We don't have Manta data to compare advection of scalar and velocity. 
// Therefore we compare to the original FluidNet code, in "init.cpp".
// TODO (aalgua): For now, only 2D is debugged!! We have to debug 3D.
void advectFluidNet(){
    auto && realCPU = CPU(at::kFloat);
    auto && realCUDA = CUDA(at::kFloat);

   for (int dim = 2; dim < 3; dim++){
      // Do advection using the 2 parameters and check against Manta.
      std::vector<std::string> m_list = {"maccormackFluidNet"};
      for (auto const& method: m_list){
         for (int sampleOutsideFluid = 0; sampleOutsideFluid < 1; sampleOutsideFluid++){
           std::string fn = std::to_string(dim) + "d_initial.bin";
           at::Tensor p;
           at::Tensor U;
           at::Tensor flags;
           at::Tensor density;
           bool is3D;

           loadMantaBatch(fn, p, U, flags, density, is3D);
           U = U.toType(realCPU);
           p = p.toType(realCPU);
           flags = flags.toType(realCPU);
           density = density.toType(realCPU);

           at::Tensor s_dst = at::zeros_like(density);
           at::Tensor s_dst_old = at::zeros_like(density);
           at::Tensor U_dst = at::zeros_like(U);
           at::Tensor U_dst_old = at::zeros_like(U);

           // Use the following tensors to rise variables from deeper levels of 
           // fluidnet code to the this high level test function.
           // You can go deeper modifying functions in tfluids.cpp and others cpp files
           // so that they take in/out tensors as references. This is useful to
           // debug and compare to our own implementation.
           at::Tensor temp = realCPU.zeros({U.size(0),3,U.size(2),U.size(3),U.size(4)});
           at::Tensor temp_CUDA = realCUDA.zeros({U.size(0),3,U.size(2),U.size(3),U.size(4)});

           float dt = 0.1;
           const int bnd = 1;
           at::Tensor maccormack_strength = realCPU.rand({}); // default in [0,1]
           float str = at::Scalar(maccormack_strength).toFloat();
           assertNotAllEqual(p);
           at::Tensor density_CUDA = density.clone().toBackend(at::Backend::CUDA);
           at::Tensor U_CUDA = U.clone().toBackend(at::Backend::CUDA);
           at::Tensor flags_CUDA = flags.clone().toBackend(at::Backend::CUDA);
           at::Tensor s_dst_CUDA = s_dst.clone().toBackend(at::Backend::CUDA);
           at::Tensor U_dst_CUDA = U_dst.clone().toBackend(at::Backend::CUDA);

           tfluids_(Main_advectScalar)(dt, flags, U, density, false, s_dst, temp,
                   method, bnd, sampleOutsideFluid, str);
           tfluids_(Main_advectVel)(dt, flags, U, false,  U_dst, temp, method, bnd, str);
           fluid::advectScalar(dt, density_CUDA, U_CUDA, flags_CUDA, s_dst_CUDA, method,
                1, sampleOutsideFluid, str);
           fluid::advectVel(dt, U_CUDA, flags_CUDA, U_dst_CUDA, method, bnd,  str);

           at::Tensor err = U_dst.toType(realCPU) - U_dst_CUDA.toType(realCPU);
           at::Tensor err_s = s_dst.toType(realCPU) - s_dst_CUDA.toType(realCPU);
           
           float err_abs = at::Scalar(at::max(at::abs(err))).toFloat();
           float precision = 1e-6; 

           std::string ss = "Test " + std::to_string(dim) + "d " + 
             ": FAILED (max error is " + std::to_string(err_abs) + ").";
           const char *css = ss.c_str();
           AT_ASSERT(err_abs < precision, css);

           float err_abs_s = at::Scalar(at::max(at::abs(err_s))).toFloat();

           ss = "Test " + std::to_string(dim) + "d " + 
             ": FAILED (max error is " + std::to_string(err_abs) + ").";
           css = ss.c_str();
           AT_ASSERT(err_abs_s < precision, css);
          }
      }
   }
   std::cout << "Advect ----------------------- [PASS]" << std::endl;
}

void createJacobianTestData(at::Tensor& p, at::Tensor& U, at::Tensor& flags,
                             at::Tensor& density) {
   auto && Tundef = at::getType(at::Backend::Undefined, at::ScalarType::Undefined);

   bool is3D = false;
   int zStart = 0;
   int zSize;
 
   if (U.size(1) == 3) {
      is3D = true;
      zSize = 4; }
   else { 
      zSize = U.size(2); }

   int bsize = std::min((int) U.size(0), (int) 3);
   
   if (p.type() != Tundef) {
      p = p.narrow(2, zStart, zSize).contiguous();
      p = p.narrow(0, 0, bsize).contiguous();
   }
   U = U.narrow(2, zStart, zSize).contiguous();
   U = U.narrow(0, 0, bsize).contiguous();

   if (flags.type() != Tundef) {
      flags = flags.narrow(2, zStart, zSize).contiguous();
      flags = flags.narrow(0, 0, bsize).contiguous();
      
      if (!(at::Scalar(at::max(at::abs(flags - fluid::TypeFluid))).toFloat() > 0)) {
        AT_ERROR("All cells are fluid.");
      } 
   }
   if (density.type() != Tundef) {
      density = p.narrow(2, zStart, zSize).contiguous();
      density = p.narrow(0, 0, bsize).contiguous();
   }

   
}                         

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
  for (int dim = 2; dim < 3; dim++){
    testSetWallBcs(dim, "advect.bin", "setWallBcs1.bin");
    testSetWallBcs(dim, "gravity.bin", "setWallBcs2.bin");
    testSetWallBcs(dim, "solvePressure.bin", "setWallBcs3.bin");
  }
   std::cout << "Set Wall Bcs ----------------------- [PASS]" << std::endl;
}

void velocityDivergence() {
   for (int dim = 2; dim < 4; dim++){
      std::string fn = std::to_string(dim) + "d_gravity.bin";
      at::Tensor undef1;
      at::Tensor U;
      at::Tensor flags;
      bool is3D;
      loadMantaBatch(fn, undef1, U, flags, undef1, is3D);
      assertNotAllEqual(U);
      assertNotAllEqual(flags);
      AT_ASSERT(is3D == (dim == 3), "Failed assert is3D");
      
      fn = std::to_string(dim) + "d_makeRhs.bin";
      at::Tensor undef2;
      at::Tensor divManta;
      at::Tensor UManta;
      at::Tensor flagsManta;
      loadMantaBatch(fn, divManta, UManta, flagsManta, undef2, is3D);
      AT_ASSERT(is3D == (dim == 3), "Failed assert is3D");
      AT_ASSERT(U.equal(UManta), "Velocity changed!");
      AT_ASSERT(flags.equal(flagsManta), "Flags changed!");   

      //Own divergence calculation.
      at::Tensor divOurs = at::zeros_like(flagsManta);
      at::Tensor u_copy = U.clone();
      at::Tensor flags_copy = flags.clone();

      fluid::velocityDivergenceForward(u_copy, flags_copy, divOurs);

      at::Tensor err = divManta - divOurs;

      float err_abs = at::Scalar(at::max(at::abs(err))).toFloat();
      float precision = 1e-6;

      std::string ss = "Test " + std::to_string(dim) +
         "d Velocity Divergence: FAILED (max error is " + std::to_string(err_abs)
         + ").";
      const char *css = ss.c_str();
      AT_ASSERT(err_abs < precision, css);

      
   }
   std::cout << "Velocity Divergence ----------------------- [PASS]" << std::endl;

}

void velocityUpdate() {
   for (int dim = 2; dim < 4; dim++){
      std::string fn = std::to_string(dim) + "d_gravity.bin";
      at::Tensor undef1;
      at::Tensor U;
      at::Tensor flags;
      bool is3D;
      loadMantaBatch(fn, undef1, U, flags, undef1, is3D);
      assertNotAllEqual(U);
      assertNotAllEqual(flags);

      AT_ASSERT(is3D == (dim == 3), "Failed assert is3D");
      fn = std::to_string(dim) + "d_correctVelocity.bin";
      at::Tensor undef2;
      at::Tensor pressure;
      at::Tensor UManta;
      at::Tensor flagsManta;
      loadMantaBatch(fn, pressure, UManta, flagsManta, undef2, is3D);
      AT_ASSERT(is3D == (dim == 3), "Failed assert is3D");
      
      AT_ASSERT(flags.equal(flagsManta), "Flags changed!");   
      AT_ASSERT(at::Scalar(at::max(at::abs(U - UManta))).toFloat() > 1e-5, "No velocities changed in Manta!");
      
      //Own velocity update calculation.
      at::Tensor UOurs = U.clone(); // This is the divergent U;
      fluid::velocityUpdateForward(UOurs, flags, pressure);
      
      at::Tensor err = UManta - UOurs;

      float err_abs = at::Scalar(at::max(at::abs(err))).toFloat();
      float precision = 1e-6;

      std::string ss = "Test " + std::to_string(dim) + 
         "d Velocity Update Forward: FAILED (max error is " + std::to_string(err_abs)
         + ").";
      const char *css = ss.c_str();
      AT_ASSERT(err_abs < precision, css);
      
   }
   std::cout << "Velocity Update Forward ----------------------- [PASS]" << std::endl;

}
at::Tensor getGravity(int dim, at::Tensor& flags) {
   float gStrength = fluid::getDx(flags) / 4;
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

void solveLinearSystemJacobi() {
   for (int dim = 2; dim < 3; dim++){
      std::string fn = std::to_string(dim) + "d_setWallBcs2.bin";
      at::Tensor undef1;
      at::Tensor UDiv;
      at::Tensor flagsDiv;
      bool is3DDiv;
      loadMantaBatch(fn, undef1, UDiv, flagsDiv, undef1, is3DDiv);
      assertNotAllEqual(UDiv);
      assertNotAllEqual(flagsDiv);
      
      fn = std::to_string(dim) + "d_solvePressure.bin";
      at::Tensor pManta;
      at::Tensor UManta;
      at::Tensor flags;
      at::Tensor rhsManta;
      bool is3D;
      loadMantaBatch(fn, pManta, UManta, flags, rhsManta, is3D);

      AT_ASSERT(at::Scalar(at::max(at::abs(flagsDiv - flags))).toFloat() == 0,
                "Flags changed!");      
      AT_ASSERT(is3D == (dim == 3), "3D boolean is inconsistent");
      AT_ASSERT(is3D == is3DDiv, "3D boolean is inconsistent (before/after solve)");

      at::Tensor div = flags.clone();
      fluid::velocityDivergenceForward(UDiv, flags, div);
     
      float precision = 1e-6;

      AT_ASSERT(at::Scalar(at::max(at::abs(rhsManta - div))).toFloat() < precision,
                "Jacobi: our divergence (rhs) is wrong!");
   
      //Note: no need to call setWallBcs as the test data already has had this
      //called.

      //Call the forward function. 
      
      at::Tensor p = flags.clone().uniform_(0,1);
      float pTol = 0;
      int maxIter = 3000; // It has VERY slow convergence. Run for long time.
      bool verbose = false;
      float residual = fluid::solveLinearSystemJacobi(
         p, flags, div, (dim == 3), pTol, maxIter, verbose);      
      std::string ss = "For " + std::to_string(dim) + "d Jacobi residual (=" + std::to_string(residual) + ") too high";
      const char *css = ss.c_str();
      AT_ASSERT(residual < 1e-4, css);
      
      p = p.toType(CPU(at::kFloat));
      float pPrecision = 1e-4;
      if (dim == 3) {
         pPrecision = 1e-3;
      }

      // Now calculate the velocity update using this new pressure and the 
      // subsequent divergence.
      at::Tensor UNew = UDiv.clone();
      fluid::velocityUpdateForward(UNew, flags, p);
      at::Tensor UDivNew = flags.clone().uniform_(0,1);
      fluid::velocityDivergenceForward(UNew, flags, UDivNew);

      ss =  "For " + std::to_string(dim) + "d Jacobi divergence error after velocityUpdate";
      css = ss.c_str();
      
      AT_ASSERT(at::Scalar(at::max(at::abs(UDivNew))).toFloat() < 1e-5, css);
      ss =  "For " + std::to_string(dim) + "d Jacobi velocity error after velocityUpdate";
      css = ss.c_str();
      AT_ASSERT(at::Scalar(at::max(at::abs(UNew - UManta))).toFloat() < 1e-4, css);
  }
   std::cout << "Solve Linear System Jacobi------------------- [PASS]" << std::endl;

}

int main(){

advectFluidNet();
setWallBcs();
velocityDivergence();
velocityUpdate();
solveLinearSystemJacobi();
addBuoyancy();
addGravity();
}
