#include <cmath>
#include <vector>

#include "ATen/ATen.h"

#include "fluid.h"

template <typename TBase>
void createPlumeBCs(std::vector<TBase>& batch, float densityVal, float uScale,
             float rad) {
  // batch at input = {p, UDiv, flags, density} 
  TBase* UDiv = batch[1];
  TBase* density = batch[3];
  TBase UBC = (UDiv->clone()).fill_(0);
  TBase UBCInvMask = (UDiv->clone()).fill_(1);
  
  TBase densityBC = (density->clone()).fill_(0);
  TBase densityBCInvMask = (density->clone()).fill_(1);

  AT_ASSERT(UBC.dim() == 5, "UBC must have dim = 5");
  AT_ASSERT(UBC.size(0) == 1, "Only single batch alllowed.");

  const int32_t xdim = UBC.size(4);
  const int32_t ydim = UBC.size(3);
  const int32_t zdim = UBC.size(2);
  const bool is3D = (UBC.size(1) == 3);
  if (!is3D) {
    AT_ASSERT(zdim == 1, "For 2D, zdim must be 1");
  } 
  float centerX = 0;//std::floor(xdim / 2); 
  float centerZ = 0;//std::max(std::floor(zdim / 2), 1); 
  float plumeRad = 1;//std::floor(xdim * rad);
  float y = 1;
  TBase vec;
  if (!is3D) {
    vec = infer_type(UBC).arange(0,2);
  }
  else {
    vec = infer_type(UBC).arange(0,3);
    vec[2] = 0;
  }
  vec.mul_(uScale);
  for (int z=0; z < zdim; z++) {
    for (int y=0; y < 4; y++) {
      for (int x=0; x < xdim; x++) {
        float dx = centerX - x;
        float dz = centerZ - z;
        if ((dx*dx + dz*dz) <= plumeRad*plumeRad) {
          // Inside the plume. Set the BCs.
          //UBC.select(1,0)[z][y][x].f
        }
      }
    }
  }
 
  batch.insert(batch.end(), std::move(UBC));
  batch.insert(batch.end(), std::move(UBCInvMask));

}

namespace fluid {
void emptyDomain(at::Tensor& tensor_flags, unsigned int bnd = 1) {
  AT_ASSERT(bnd != 0, "bnd must be non zero!");
  AT_ASSERT(tensor_flags.dim() == 5, "Flags should be 5D");
  AT_ASSERT(tensor_flags.size(1) == 1, "Flags should be a scalar");

  const bool is_3d = (tensor_flags.size(2) != 1);

  AT_ASSERT(((!is_3d || tensor_flags.size(2) > bnd * 2) &&
             tensor_flags.size(3) > bnd * 2 || tensor_flags.size(4) > bnd * 2),
             "Simulation domain is not big enough");

  const int32_t xsize  = tensor_flags.size(4);
  const int32_t ysize  = tensor_flags.size(3);
  const int32_t zsize  = tensor_flags.size(2);
  const int32_t nbatch = tensor_flags.size(0);

  //tensor_flags.select(1,0) = TypeFluid;
  for (int b_it = 0; b_it < bnd; b_it++){
    if (is_3d) {
      tensor_flags.select(2,b_it)         = TypeObstacle;
      tensor_flags.select(2,xsize-1-b_it) = TypeObstacle;
    }
    tensor_flags.select(3, b_it)         = TypeObstacle;
    tensor_flags.select(3, ysize-1-b_it) = TypeObstacle;
    tensor_flags.select(4, b_it)         = TypeObstacle;
    tensor_flags.select(4, xsize-1-b_it) = TypeObstacle;
  }

//  fluid::FlagGrid flags(tensor_flags, is_3d);
//
//  const int32_t xsize = flags.xsize();
//  const int32_t ysize = flags.ysize();
//  const int32_t zsize = flags.zsize();
//  const int32_t nbatch  = flags.nbatch();
//  for (int32_t b = 0; b < nbatch; b++) {
//    int32_t k, j, i;
//#pragma omp parallel for collapse(3) private(k, j, i)
//    for (k = 0; k < zsize; k++) {
//      for (j = 0; j < ysize; j++) {
//        for (i = 0; i < xsize; i++) {
//          if (i < bnd || i > xsize - 1 - bnd ||
//              j < bnd || j > ysize - 1 - bnd ||
//              (is_3d && (k < bnd || k > zsize - 1 - bnd))) {
//            flags(i, j, k, b) = TypeObstacle;
//          } else {
//            flags(i, j, k, b) = TypeFluid;
//          }
//        }
//      }
//    }
//  }

} 
} // namespace fluid

int main() {

  auto && Tfloat = CPU(at::kFloat);

  int res = 10;
  int batchSize = 1;

  at::Tensor pDiv    = Tfloat.zeros({batchSize, 1, 1, res, res});     
  at::Tensor UDiv    = Tfloat.zeros({batchSize, 2, 1, res, res});
  at::Tensor flags   = Tfloat.zeros({batchSize, 1, 1, res, res});
  at::Tensor density = Tfloat.zeros({batchSize, 1, 1, res, res});
 
  fluid::emptyDomain(flags, 3); 


  at::Tensor y = Tfloat.rand({6,16,10});
  bool is3D = (y.size(0) > 1);

  at::Tensor rhs = Tfloat.rand_like(y);
  int d = y.size(0);
  int w = y.size(1);
  int h = y.size(2);

  std::cout << y << std::endl;

  at::Tensor index_x = CPU(at::kLong).arange(0, h).view({h}).expand_as(y);
  at::Tensor index_y = CPU(at::kLong).arange(0, w).view({w, 1}).expand_as(y);
  at::Tensor index_z;
  if (is3D) {
     index_z = CPU(at::kLong).arange(0, d).view({d, 1 , 1}).expand_as(y);
  }
  //std::cout << index_x << std::endl;
  //std::cout << index_y << std::endl;
  //std::cout << index_z << std::endl;
  //at::Tensor index_ten = index_x.expand_as(y);
  //at::Tensor index_ten = CPU(at::kInt).arange(1,h * w + 1).view({h, w});
  at::Tensor index_ten;
  
  if (!is3D) {
    index_ten = at::stack({index_x, index_y}, 0);
  }
  else { 
    index_ten = at::stack({index_x, index_y, index_z}, 0);
  }
  int bnd_n = 1;

  at::Tensor mask = (index_ten.select(0,0) < bnd_n).__or__(index_ten.select(0,0) > h - 1 - bnd_n)
                    .__or__(index_ten.select(0,1) < bnd_n).__or__(index_ten.select(0,1) > w - 1 - bnd_n);
  std::cout << mask << std::endl;
  if (is3D) {
      mask = mask.__or__(index_ten.select(0,2) < bnd_n).__or__(index_ten.select(0,2) > d - 1 - bnd_n);
  } 
  
  rhs.masked_fill_(mask, 0);
  std::cout << rhs << std::endl;

  //at::Tensor mask = z.ge(0.5);
  //mask[0] = 1;
  //at::Tensor mask_y = z.ge(0.1).toType(y.type());
  //std::cout << mask_y << std::endl;
  //std::cout << x << std::endl;
  //std::cout << mask << std::endl; 
  //std::cout << mask.nonzero() << std::endl;

//get row indices
  //at::Tensor indices1 = mask.nonzero().squeeze(1);
  //std::cout << indices1 << std::endl;

  //at::Tensor index_select = at::index(z, {{},mask,{}});
  //std::cout << y << std::endl;
  //std::cout << z << std::endl;
  //std::cout << index_select << std::endl;
  //at::Tensor indices2 = at::index_put_(y, {{}, mask,{}}, index_select);
  //std::cout << indices2 << std::endl;  
  //std::cout << z.view({1,2,1,3}) << std::endl;

  at::Tensor densityVal = CPU(at::kFloat).ones({1});
  float rad = 0.15;
  
  
}
