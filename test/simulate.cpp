#include <cmath>
#include <vector>
#include <sstream>
#include <omp.h>

#include "load_manta_data.h"
#include "plot_utils.h"

#include "ATen/ATen.h"

#include "fluid.h"

void createPlumeBCs(std::vector<at::Tensor>& batch, float densityVal, float uScale,
             float rad) {
  // batch at input = {p, UDiv, flags, density} 

  AT_ASSERT(batch.size() == 4, "Batch must have size 4");
  at::Tensor UDiv = batch[1];
  at::Tensor density = batch[3];
  at::Tensor UBC = UDiv.clone().fill_(0);
  at::Tensor UBCInvMask = UDiv.clone().fill_(1);

  // Single density value  
  at::Tensor densityBC = density.clone().fill_(0);
  at::Tensor densityBCInvMask = density.clone().fill_(1);

  AT_ASSERT(UBC.dim() == 5, "UBC must have dim = 5");
  AT_ASSERT(UBC.size(0) == 1, "Only single batch alllowed.");

  const int32_t xdim = UBC.size(4);
  const int32_t ydim = UBC.size(3);
  const int32_t zdim = UBC.size(2);
  const bool is3D = (UBC.size(1) == 3);
  if (!is3D) {
    AT_ASSERT(zdim == 1, "For 2D, zdim must be 1");
  } 
  float centerX = std::floor( (float) xdim / 2); 
  float centerZ = std::max(std::floor( (float) zdim / 2), (float) 1); 
  float plumeRad = std::floor( (float) xdim * (float) rad);

  float y = 1;
  at::Tensor vec;
  if (!is3D) {
    vec = infer_type(UBC).arange(0,2);
  }
  else {
    vec = infer_type(UBC).arange(0,3);
    vec[2] = 0;
  }
  vec.mul_(uScale);

  at::Tensor index_x = infer_type(UDiv).arange(0, xdim).view({xdim}).expand_as(density[0][0]);
  at::Tensor index_y = infer_type(UDiv).arange(0, ydim).view({ydim, 1}).expand_as(density[0][0]);
  at::Tensor index_z;
  if (is3D) {
     index_z = CPU(at::kFloat).arange(0, zdim).view({zdim, 1 , 1}).expand_as(density[0][0]);
  }
  at::Tensor index_ten;

  if (!is3D) {
    index_ten = at::stack({index_x, index_y}, 0);
  }
  else { 
    index_ten = at::stack({index_x, index_y, index_z}, 0);
  }
 
  //TODO 3d implementation
  at::Tensor indx_circle = index_ten.narrow(2, 0, 4);
  indx_circle.select(0,0) -= centerX;
  at::Tensor maskInside = (indx_circle.select(0,0).pow(2) <= plumeRad*plumeRad);

  // Inside the plume. Set the BCs.

  //It is clearer to just multiply by mask (casted into Float)
  at::Tensor maskInside_f = maskInside.type().toScalarType(density.type().scalarType()).
                              copy(maskInside);
  UBC.narrow(3,0,4) = maskInside_f * vec.view({1,2,1,1,1}).expand_as(UBC.narrow(3,0,4));
  UBCInvMask.narrow(3,0,4).masked_fill_(maskInside, 0);

  densityBC.narrow(3,0,4).masked_fill_(maskInside, densityVal);
  densityBCInvMask.narrow(3,0,4).masked_fill_(maskInside, 0);

  // Outside the plume. Set the velocity to zero and leave density alone.  

  at::Tensor maskOutside = (maskInside == 0);
  UBC.narrow(3,0,4).masked_fill_(maskOutside, 0);
  UBCInvMask.narrow(3,0,4).masked_fill_(maskOutside, 0);
  
  // Insert the new tensors at the end of the batch. 
  batch.insert(batch.end(), std::move(UBC));
  batch.insert(batch.end(), std::move(UBCInvMask));

  batch.insert(batch.end(), std::move(densityBC));
  batch.insert(batch.end(), std::move(densityBCInvMask));

  // batch at output = {0: p, 1: UDiv, 2: flags, 3: density, 4: UBC,
  //                       5: UBCInvMask, 6: densityBC, 7: densityBCInvMask} 
}

void setConstVals(std::vector<at::Tensor>& batch, at::Tensor& p,
                  at::Tensor& U, at::Tensor& flags, at::Tensor& density) {
 // apply external BCs.
 // batch  = {0: p, 1: UDiv, 2: flags, 3: density, 4: UBC,
 //           5: UBCInvMask, 6: densityBC, 7: densityBCInvMask} 

 // Zero out the U values on the BCs.
 U.mul_(batch[5]);
 // Add back the values we want to specify.
 U.add_(batch[4]);

 density.mul_(batch[7]);
 density.add_(batch[6]);

}

namespace fluid {

float getDx(at::Tensor flags) {
  float gridSizeMax = std::max(std::max(flags.size(2), flags.size(3)), flags.size(4));
  return (1.0 / gridSizeMax);
}

void emptyDomain(at::Tensor& flags, int bnd = 1) {
  AT_ASSERT(bnd != 0, "bnd must be non zero!");
  AT_ASSERT(flags.dim() == 5, "Flags should be 5D");
  AT_ASSERT(flags.size(1) == 1, "Flags should be a scalar");

  const bool is3D = (flags.size(2) != 1);

  AT_ASSERT(((!is3D || flags.size(2) > bnd * 2) &&
             flags.size(3) > bnd * 2 || flags.size(4) > bnd * 2),
             "Simulation domain is not big enough");

  const int32_t xdim  = flags.size(4);
  const int32_t ydim  = flags.size(3);
  const int32_t zdim  = flags.size(2);
  const int32_t nbatch = flags.size(0);

  at::Tensor index_x = infer_type(flags).arange(0, xdim).view({xdim}).expand_as(flags[0][0]);
  at::Tensor index_y = infer_type(flags).arange(0, ydim).view({ydim, 1}).expand_as(flags[0][0]);
  at::Tensor index_z;
  if (is3D) {
     index_z = infer_type(flags).arange(0, zdim).view({zdim, 1 , 1}).expand_as(flags[0][0]);
  }
  at::Tensor index_ten;
  if (!is3D) {
    index_ten = at::stack({index_x, index_y}, 0);
  }
  else {
    index_ten = at::stack({index_x, index_y, index_z}, 0);
  }

  at::Tensor maskBorder = (index_ten.select(0,0) < bnd).__or__
                          (index_ten.select(0,0) > xdim - 1 - bnd).__or__
                          (index_ten.select(0,1) < bnd).__or__
                          (index_ten.select(0,1) > ydim - 1 - bnd);
  if (is3D) {
      maskBorder = maskBorder.__or__(index_ten.select(0,2) < bnd).__or__
                                    (index_ten.select(0,2) > zdim - 1 - bnd);
  }

  flags.masked_fill_(maskBorder, fluid::TypeObstacle);
  flags.masked_fill_((maskBorder == 0), fluid::TypeFluid); 
}

// Top level simulation loop.
void simulate(std::vector<at::Tensor>& batch, int res) {
  
  float dt = 0.1;
  float maccormackStrength = 0.6; 
  bool sampleOutsideFluid = false;

  float buoyancyScale = 2.0 * (res / 128);
  float gravityScale = 1 * (res / 128);
  
  // Get p, U, flags and density from batch.
  at::Tensor p = batch[0];
  at::Tensor U = batch[1];
  at::Tensor flags = batch[2];
  at::Tensor density = batch[3];

  // First advect all scalar fields.
  at::Tensor density_dst = density.clone();
  fluid::old::advectScalar(dt, flags, U, density, true,  density_dst, "maccormackFluidNet",
                      1, sampleOutsideFluid, maccormackStrength);
  std::cout << "density" << std::endl;
  std::cout << density << std::endl;

//  density = density_dst;
  // Self-advect velocity
  at::Tensor vel_dst = U.clone();
  fluid::old::advectVel(dt, flags, U, true, vel_dst,  "maccormackFluidNet", 1, maccormackStrength);
 // U = vel_dst;

  // Set the manual BCs.
  setConstVals(batch, p, U, flags, density);

  at::Tensor gravity = infer_type(U).tensor({3}).fill_(0);  
  gravity[1] = 1;

  // Add external forces: buoyancy.
  gravity.mul_(-(fluid::ten::getDx(flags) / 4) * buoyancyScale);

  fluid::addBuoyancy(U, flags, density, gravity, dt);
  
  // Add external forces: gravity.
  gravity.mul_((-fluid::ten::getDx(flags) / 4) * gravityScale);
  fluid::addGravity(U, flags, gravity, dt);

  // Set the constant domain values.
  fluid::setWallBcsForward(U, flags);
  setConstVals(batch, p, U, flags, density);

  at::Tensor div = p.clone();
  fluid::velocityDivergenceForward(U, flags, div);

  bool is3D = (U.size(2) == 3);
  float residual;
  float pTol = 0;
  int maxIter = 34;

  // Until replaced by a tensorial version, it must be done in CPU.
 
  residual = fluid::solveLinearSystemJacobi(p, flags, div, is3D, pTol, maxIter, false);
  

  fluid::velocityUpdateForward(U, flags, p);
  setConstVals(batch, p, U, flags, density);
}

} // namespace fluid

int main() {

int res = 10;

at::Tensor p =       CPU(at::kFloat).zeros({1,1,1,res,res});
at::Tensor U =       CPU(at::kFloat).zeros({1,2,1,res,res});
at::Tensor flags =   CPU(at::kFloat).zeros({1,1,1,res,res});
at::Tensor density = CPU(at::kFloat).zeros({1,1,1,res,res});

fluid::emptyDomain(flags);
std::vector<at::Tensor> batch;

batch.insert(batch.end(), p);
batch.insert(batch.end(), U);
batch.insert(batch.end(), flags);
batch.insert(batch.end(), density);

float densityVal = 1;
float rad = 0.15;
float plumeScale = 1.0 * ((float) res/128);

createPlumeBCs(batch, densityVal, plumeScale, rad);
int maxIter = 2;
int outIter = 100000;
int it = 0;
while (it < maxIter) {
  std::cout << "Iteration " << it+1 << " out of " << maxIter << std::endl;
  fluid::simulate(batch, res);
  std::cout << "density" << std::endl;
  std::cout << batch[3] << std::endl;
  it++;
  if (it % outIter == 0) {
    std::cout << "Writing output at iteration " << it << std::endl;
    std::string name_density = "density_it_" + std::to_string(it);
    plotTensor2D(batch[3].toBackend(at::Backend::CPU), 500, 500, name_density);
  }
}
}
