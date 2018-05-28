#include <sstream>
#include <cassert>
#include <tuple>

#include "ATen/ATen.h"

#include "type_test.h"
#include "load_manta_data.h"
#include "plot_utils.h"
#include "fluid.h"

using namespace at;
using namespace fluid;
using namespace cv;

void advectFluidNet(){
   for (int dim = 2; dim < 4; dim++){
      // Do advection using the 2 parameters and check against Manta.
      std::vector<std::string> m_list = {"eulerFluidNet", "maccormackFluidNet"};
      for (auto const& method: m_list){
         for (int sampleOutsideFluid = 0; sampleOutsideFluid < 2; sampleOutsideFluid++){
           std::string fn = std::to_string(dim) + "d_initial.bin";
           at::Tensor p;
           at::Tensor U;
           at::Tensor flags;
           at::Tensor density;
           bool is3D;
           
           loadMantaBatch(fn, p, U, flags, density, is3D);
           // ASSERT IS 3D

           float dt = 0.1;
           int boundaryWidth = 0;
           at::Tensor maccormackStrength = CPU(at::kFloat).rand({1}); // default in [0,1]
           assertNotAllEqual(p);
         
         }
      } 
   } 
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
      
      if (!(at::Scalar(at::max(at::abs(flags - TypeFluid))).toFloat() > 0)) {
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

   if (!(at::Scalar(at::max(at::abs(flags - TypeFluid))).toFloat() > 0)) {
      AT_ERROR("All cells are fluid at input.");
   }

   fn = std::to_string(dim) + "d_" + fnOutput;
   at::Tensor UManta;
   at::Tensor undef2;
   at::Tensor flagsManta;
   loadMantaBatch(fn, undef2, UManta, flagsManta, undef2, is3D);

   if (!(is3D == (dim == 3))) {
      AT_ERROR("Error with is3D at output.");}
   
   if (!(flags.equal(flagsManta))) {
      AT_ERROR("Flags changed!"); }

   at::Tensor UOurs = U.clone();

   fluid::setWallBcsForward(flags, UOurs, is3D);
   at::Tensor err = UOurs - UManta;
    
   float err_abs = at::Scalar(at::max(at::abs(err))).toFloat();
   float precision = 1e-6; 

   if (err_abs < precision) {
      std::cout << "Test " << dim << "d Set Wall BCs Forward: OK!" << std::endl;
   }
   else {
   std::cout << "Test " << dim << "d Set Wall BCs Forward: FAILED (max error is " 
   << err_abs << ")."  << std::endl;
   }
}

void setWallBcs() {
  for (int dim = 2; dim < 4; dim++){
    testSetWallBcs(dim, "advect.bin", "setWallBcs1.bin");
  }  
}

int main(){

auto && Tfloat = CPU(kFloat);
auto && Tint   = CPU(kInt);
auto && Tbyte  = CPU(kByte);

int nbatch = 1;
int bnd    = 1;
int depth  = 1;
int height = 3;
int width  = 4;

Tensor flagsGT = Tfloat.ones({nbatch, 1, depth, height, width});
Tensor vel =     Tfloat.rand({nbatch, 3, depth, height, width}) * 10;
Tensor rho =     Tfloat.zeros({nbatch, 1, depth, height, width});
Tensor rho_dst = Tfloat.zeros({nbatch, 1, depth, height, width});
Tensor fwd =     Tfloat.zeros({nbatch, 1, depth, height, width});
Tensor bwd =     Tfloat.zeros({nbatch, 1, depth, height, width});
Tensor fwd_pos = Tfloat.zeros({nbatch, 3, depth, height, width});
Tensor bwd_pos = Tfloat.zeros({nbatch, 3, depth, height, width});

Tensor fwd_u =   Tfloat.zeros({nbatch, 3, depth, height, width});
Tensor bwd_u =   Tfloat.zeros({nbatch, 3, depth, height, width});
Tensor vel_dst = Tfloat.zeros({nbatch, 3, depth, height, width});

Tensor p =            Tfloat.zeros({nbatch, 1, depth, height, width});     
Tensor div =          Tfloat.zeros({nbatch, 1, depth, height, width});
Tensor div_test =     Tfloat.zeros({nbatch, 1, depth, height, width});
Tensor p_prev =       Tfloat.rand({nbatch, 1, depth, height, width});
Tensor p_delta =      Tfloat.zeros({nbatch, 1, depth, height, width});
Tensor p_delta_norm = Tfloat.zeros({nbatch, 1, depth, height, width});

//vel.select(1,0) = 1;
//vel.select(1,2) = 0;

flagsGT.select(3, 0) = 2;
flagsGT.select(3, height-1) = 2;
flagsGT.select(4,0) = 2;
flagsGT.select(4, width-1) = 2;
//
//flagsGT[0][0][0][2][3] = TypeObstacle;

//vel.select(1,0) = 1;
//vel.select(1,1) = 1;
//vel.select(4,0) = 0;
//vel.select(4, width-1) = 0;
//vel.select(3, 0) = 0;
//vel.select(3, height-1) = 0;
//std::cout << "Flag grid: " << std::endl;
//std::cout << flagsGT << std::endl;
//std::cout << "\n";

FlagGrid flag(flagsGT, false);

//vel *= 2.14;
//std::cout << "Vel before: " << std::endl;
//std::cout << vel << std::endl;
//std::cout << "\n";
std::cout << "Setting wall boundary conditions..." << std::endl;
setWallBcsForward(flagsGT, vel, false);
std::cout << "\n";
//std::cout << "Vel after" << std::endl;
//std::cout << vel << std::endl;
//std::cout << "\n";

std::cout << "Advection of scalar ..." << std::endl;
const std::string method = "maccormackFluidNet";

std::tuple<float, T&, T&, T&, T&, T&, T&, T&, T&,
           bool, std::string, int32_t, bool, float>
             args{1., flagsGT, vel, rho, rho_dst, fwd, bwd, fwd_pos,
                  bwd_pos, false, method, 1, false, 1.};


//  test::save_it_for_later<float, T&, T&, T&, T&, T&, T&, T&, T&,
 //          bool, std::string, int32_t, bool, float> saved = {args, advectScalar};

//advectScalar(std::get<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13>(args)...);

//test::profileAndTestCuda(advectScalar, 1., flagsGT, vel, rho, rho_dst, fwd, bwd, fwd_pos,
//                   bwd_pos, false, "maccormackFluidNet", 1, false, 1);
//advectScalar(1., flagsGT, vel, rho, rho_dst, fwd, bwd, fwd_pos,
//             bwd_pos, false, "maccormackFluidNet", 1, false, 1); 
std::cout << "\n";
std::cout << "Advection of velocity ..." << std::endl;
advectVel(1., flagsGT, vel, vel_dst, fwd_u, bwd_u,
          false, "maccormackFluidNet", 1, 1); 
std::cout << "\n";

//std::cout << "Initial rho : " << std::endl;
//std::cout << rho << std::endl;
//std::cout << "\n";
//
//std::cout << "Advected Scalar field : " << std::endl;
//std::cout << rho_dst << std::endl;
//std::cout << "\n";

std::cout << "Divergence calculation ..." << std::endl;
velocityDivergenceForward(flagsGT, vel, div, false);
std::cout << "\n";

std::cout << "Solving Linear System ..." << std::endl;
solveLinearSystemJacobi(&p, &flagsGT, &div, &p_prev, &p_delta, &p_delta_norm,
                        false, 0.0001, 34, false);
std::cout << "\n";

//std::cout << "Pressure : " << std::endl;
//std::cout << p << std::endl;
//std::cout << "\n";
//
std::cout << "Update velocity forward ..." << std::endl;
velocityUpdateForward(flagsGT, vel, p, false);
std::cout << "\n";
//
velocityDivergenceForward(flagsGT, vel, div_test, false);
//std::cout << "Final divergence : " << std::endl;
//std::cout << div_test << std::endl;
//std::cout << "\n";

//std::cout << "Advected scalar field : " << std::endl;
//std::cout << rho_dst <<std::endl;
//std::cout << "\n";
//std::cout << "Forward pos : " << std::endl;
//std::cout << fwd_pos << std::endl; 
//std::cout << "\n";

Tensor p_read;
Tensor U_read;
Tensor flags_read;
Tensor density_read;
bool is3D;

//plotTensor2D(p_read, 517, 517);
//plotTensor2D(U_read.select(1,0), 516, 517);
//plotTensor2D(U_read.select(1,1), 516, 517);
//plotTensor2D(density_read, 517, 517, "density");

//plotTensor2D(density_read.select(1,0), 516, 516);
int dim = 3;
std::string fn = std::to_string(dim) + "d_solvePressure.bin";
std::cout << fn << std::endl;

//loadMantaBatch(fn, p_read, U_read, flags_read, density_read, is3D);
//std::cout << flags_read << std::endl;
setWallBcs();

}
