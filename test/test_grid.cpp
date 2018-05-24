#include <sstream>
#include <cassert>

#include "ATen/ATen.h"

#include "type_test.h"
#include "load_manta_data.h"
#include "plot_utils.h"
#include "fluid.h"

using namespace at;
using namespace fluid;
using namespace cv;

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
advectScalar(1., flagsGT, vel, rho, rho_dst, fwd, bwd, fwd_pos,
             bwd_pos, false, "maccormackFluidNet", 1, false, 1); 
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

T p_read;
T U_read;
T flags_read;
T density_read;
bool is3D;

bool succes = loadMantaFile("../test_data/b15_2d_advect_openBounds_False_order_1.bin", p_read, U_read, flags_read, density_read, is3D);

//plotTensor2D(p_read, 517, 517);
//plotTensor2D(U_read.select(1,0), 516, 517);
//plotTensor2D(U_read.select(1,1), 516, 517);
plotTensor2D(density_read, 517, 517, "density");

//plotTensor2D(density_read.select(1,0), 516, 516);
int dim = 2;
std::string fn = std::to_string(dim) + "d_initial.bin";
loadMantaBatch(fn);

}
