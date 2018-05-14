#include "ATen/ATen.h"
#include <sstream>
#include <cassert>

#include "type_test.h"
#include "grid/grid.h"
#include "advection/advection.h"

using namespace at;
using namespace fluid;
int main(){

/*int nbatch = 2;
int xsize = 10;
int ysize = 10;
int zsize = 1;
bool is3d = false;
Tensor flags = CPU(kInt).zeros({nbatch, 1, zsize, ysize, xsize});
Tensor u = CPU(kFloat).rand({nbatch, 3, zsize, ysize, xsize}); 
Tensor scalar = CPU(kFloat).rand({nbatch, 1, zsize, ysize, xsize}); 
Tensor s_dst = CPU(kFloat).zeros({nbatch, 1, zsize, ysize, xsize}); 
Tensor fwd = CPU(kFloat).zeros({nbatch, 1, zsize, ysize, xsize}); 
Tensor bwd = CPU(kFloat).zeros({nbatch, 1, zsize, ysize, xsize}); 
Tensor fwd_pos = CPU(kFloat).zeros({nbatch, 3, zsize, ysize, xsize}); 
Tensor bwd_pos = CPU(kFloat).zeros({nbatch, 3, zsize, ysize, xsize}); 
Tensor bwd_pos = CPU(kFloat).zeros({nbatch, 3, zsize, ysize, xsize}); 
std::string method = "maccormack";
const int bnd_width = 1;
const bool sample_out = false;
const float strength = 1.;

advectScalar(1., &flags, &u, &scalar, &s_dst, &fwd, &bwd, &fwd_pos, 
*/
auto && Tfloat = CPU(kFloat);
auto && Tint   = CPU(kInt);
auto && Tbyte  = CPU(kByte);

//Tensor grid = Tfloat.ones({1,1,zsize, ysize, xsize});
//Tensor scalar = Tfloat.zeros({1,1,zsize, ysize, xsize});
//Tensor v = Tfloat.ones({1,2,zsize, ysize, xsize});
//Tensor omega = Tfloat.zeros({1,2,zsize, ysize, xsize});
//
//grid.select(4,0) = 2;
//grid.select(4,grid.size(4) - 1) = 2;
//grid.select(3,0) = 2;
//grid.select(3, grid.size(3) - 1) = 2;
//
//scalar[0][0][0][1][1] = 10;
//scalar[0][0][0][1][2] = 5;
//scalar[0][0][0][2][1] = 5;
//scalar[0][0][0][2][2] = 5;
//
//v.select(4,0) = 0;
//v.select(4,grid.size(4) - 1) = 0;
//v.select(3,0) = 0;
//v.select(3, grid.size(3) - 1) = 0;
//
//FlagGrid flag(grid, false);
//RealGrid pres(scalar, false);
//MACGrid vel(v, false);
//VecGrid vort(omega, false);
//
//bool linetrace;
//
//Tensor pos = CPU(kFloat).tensor({3});
//Tensor delta = CPU(kFloat).tensor({3});
//Tensor new_pos = CPU(kFloat).tensor({3});
//float dt = 2.;
//pos[0] = 1.99;
//pos[1] = 1.5;
//pos[2] = 0.5;
//
//delta[0] = 1;
//delta[1] = 0.0001;
//delta[2] = 0;
//
//Tensor at_zero = getType(flag.getBackend(),
//                         kInt ).scalarTensor(0);
//
//std::cout << delta * dt << std::endl;
//linetrace = calcLineTrace(pos, delta, flag, at_zero , &new_pos, true);
//
//std::cout << new_pos << std::endl;
//float minv = std::numeric_limits<float>::infinity();
//T minus = CPU(kFloat).scalarTensor(minv);
//pos[2] += 1;
//std::cout << pos.clamp((int) 1, (int) 3) << std::endl;
//
//std::cout << vel(1,2,0,0) << std::endl;
//
//Tensor pos = Tfloat.zeros({3});
//pos[0] = 1;
//pos[1] = 3;
//pos[2] = 1;
//
//Tensor new_val = Tfloat.tensor({3});
//new_val[0] = 2;
//new_val[1] = 10;
//new_val[2] = 0;

//pres_a[0][0][0][0][0] = scalar[0][0][0][2][2];
//std::cout << vort << std::endl;

//for(int i = 0; i < foo_a.size(0); i++) {
//    for (int j = 0; j < foo_a.size(1); j++) {
//      // use the accessor foo_a to get tensor data.
//      std::cout << foo_a[i][j] << std::endl;
//    }
//}
//FloatGrid test_float(T, test, false);
//FlagGrid test_flag(T, flag, false);
//test.select(4,0) = 1;
//test.select(4,1) = 2;
//test.select(4,2) = 3;
//
//flag.select(3,0) = 2;
//flag.select(3,3) = 2;
//flag.select(4,0) = 2;
//flag.select(4,3) = 2;
//
//Tensor num = flag[0][0][0][0][0];
//Tensor num1 = flag[0][0][0][0][1];
//Tensor logic = num.lt(num1);
//std::cout << logic << std::endl;
//Scalar on_gpu = Scalar(flag[0][0][0][0][0]);
//assert(on_gpu.isBackedByTensor());


int nbatch = 1;
int bnd    = 1;
int depth  = 1;
int height = 5;
int width  = 4;

Tensor flagsGT = Tfloat.ones({nbatch, 1, depth, height, width});
Tensor vel =     Tfloat.zeros({nbatch, 3, depth, height, width});
Tensor rho =     Tfloat.rand({nbatch, 1, depth, height, width});
Tensor rho_dst = Tfloat.zeros({nbatch, 1, depth, height, width});
Tensor fwd =     Tfloat.zeros({nbatch, 1, depth, height, width});
Tensor bwd =     Tfloat.zeros({nbatch, 1, depth, height, width});
Tensor fwd_pos = Tfloat.zeros({nbatch, 3, depth, height, width});
Tensor bwd_pos = Tfloat.zeros({nbatch, 3, depth, height, width});


Tensor fwd_u =     Tfloat.zeros({nbatch, 3, depth, height, width});
Tensor bwd_u =     Tfloat.zeros({nbatch, 3, depth, height, width});
Tensor vel_dst = Tfloat.zeros({nbatch, 3, depth, height, width});

flagsGT.select(4,0) = 2;
flagsGT.select(4, width-1) = 2;
flagsGT.select(3, 0) = 2;
flagsGT.select(3, height-1) = 2;

vel.select(1,0) = 1;
vel.select(1,1) = 1;
vel.select(4,0) = 0;
vel.select(4, width-1) = 0;
vel.select(3, 0) = 0;
vel.select(3, height-1) = 0;




std::cout << "Domain : ";
std::cout << flagsGT << std::endl;
std::cout << "\n";
std::cout << "Initial Velocity field : " << std::endl;
std::cout << vel << std::endl;
std::cout << "\n";

std::cout << "Advection of scalar ..." << std::endl;
advectScalar(1., flagsGT, vel, rho, rho_dst, fwd, bwd, fwd_pos,
             bwd_pos, false, "maccormackFluidNet", 1, false, 1); 
std::cout << "\n";
std::cout << "Advection of velocity ..." << std::endl;
advectVel(1., flagsGT, vel, vel_dst, fwd_u, bwd_u,
          false, "maccormackFluidNet", 1, 1); 
std::cout << "\n";

std::cout << "Initial Scalar field : " << std::endl;
std::cout << rho << std::endl;
std::cout << "\n";
std::cout << "Advected scalar field : " << std::endl;
std::cout << rho_dst <<std::endl;
std::cout << "\n";
std::cout << "Forward pos : " << std::endl;
std::cout << fwd_pos << std::endl; 
std::cout << "\n";

}
