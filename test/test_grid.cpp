#include "ATen/ATen.h"
#include <sstream>
#include <cassert>

#include "type_test.h"
#include "../src/grid/grid_scalar.h"

using namespace at;

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
int xsize = 4;
int ysize = 4;
int zsize = 1;

auto && Tfloat = CPU(kFloat);
auto && Tint   = CPU(kInt);
auto && Tbyte  = CPU(kByte);

Tensor grid = Tfloat.ones({1,1,zsize, ysize, xsize});
Tensor scalar = Tfloat.zeros({1,1,zsize, ysize, xsize});
Tensor v = Tfloat.ones({1,2,zsize, ysize, xsize});
Tensor omega = Tfloat.zeros({1,2,zsize, ysize, xsize});

grid.select(4,0) = 2;
grid.select(4,grid.size(4) - 1) = 2;
grid.select(3,0) = 2;
grid.select(3, grid.size(3) - 1) = 2;

scalar[0][0][0][1][1] = 10;
scalar[0][0][0][1][2] = 5;
scalar[0][0][0][2][1] = 5;
scalar[0][0][0][2][2] = 5;

v.select(4,0) = 0;
v.select(4,grid.size(4) - 1) = 0;
v.select(3,0) = 0;
v.select(3, grid.size(3) - 1) = 0;

FlagGrid flag(grid, false);
RealGrid pres(scalar, false);
MACGrid vel(v, false);
VecGrid vort(omega, false);

Tensor chan = CPU(kFloat).arange(3);
std::cout << vel << std::endl;

for (int j=1; j < vel.ysize() - 1; j++){
   for (int i=1; i < vel.xsize() - 1; i++){
    
    vort.setSafe(i,j,0,0, vel.getCentered(i,j,0,0) );
    vort(i,j,0,0) = vort.curl(i,j,0,0);
   }
}

std::cout << vort << std::endl; 
}
/*
std::cout << vel(1,2,0,0) << std::endl;

Tensor pos = Tfloat.zeros({3});
pos[0] = 1;
pos[1] = 3;
pos[2] = 1;

Tensor new_val = Tfloat.tensor({3});
new_val[0] = 2;
new_val[1] = 10;
new_val[2] = 0;

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


std::cout << test_grid << std::endl;
std::cout << test[0][0][0][0][1] << std::endl;
std::cout << test_float.interpol(pos, 0) << std::endl;

int nbatch = 1;
int bnd    = 1;
int width  = 4;
int height = 4;
int depth  = 1;

Tensor flagsGT = zeros(T, {nbatch, 1, depth, height, width});
Tensor vel = zeros(T, {nbatch, 3, depth, height, width});
Tensor rho = rand(T, {nbatch, 1, depth, height, width});
Tensor rho_dst = zeros(T, {nbatch, 1, depth, height, width});
Tensor fwd = zeros(T, {nbatch, 1, depth, height, width});
Tensor bwd = zeros(T, {nbatch, 1, depth, height, width});
Tensor fwd_pos = zeros(T,{nbatch, 3, depth, height, width});
Tensor bwd_pos = zeros(T,{nbatch, 3, depth, height, width});

flagsGT.select(4,0) = 1;
flagsGT.select(4, height-1) = 1;
flagsGT.select(3, 0) = 1;
flagsGT.select(3, width-1) = 1;

vel.select(3,1) = 1;
//advectScalar(1., &flagsGT, &vel, &rho, &rho_dst, &fwd, &bwd, &fwd_pos,
             &bwd_pos, false, "eulerFluidNet", 1, false); 
std::cout << "Velocity field : " << std::endl;
std::cout << vel << std::endl;
std::cout << rho << std::endl;
std::cout << rho_dst <<std::endl;

Tensor foo = CPU(kFloat).rand({4,4});

// assert foo is 2-dimensional and holds floats.
auto foo_a = foo.accessor<float,2>();

for(int i = 0; i < foo_a.size(0); i++) {
    for (int j = 0; j < foo_a.size(1); j++) {
      // use the accessor foo_a to get tensor data.
      std::cout << foo_a[i][j] << std::endl;
    }
}
}
*/
