#include "grid.h"
#include "ATen/ATen.h"
#include <iostream>

int main(){
  at::Tensor t = at::CUDA(at::kFloat).rand({128, 64, 1, 3, 3});
  at::Tensor* p_t = &t;
  GridBase test(p_t,false);
  
  std::cout << "Is the data 3D? : " << test.is_3d() << "\n";
  std::cout << "Dx of the sim : " << test.getDx() << "\n";
  Int3 vec_int(4,4,5);
  std::cout << "Test isInBounds : " << test.isInBounds(vec_int, 10) 
       << "\n";




}
