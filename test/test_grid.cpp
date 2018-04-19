#include "grid.h"
#include "ATen/ATen.h"
#include <iostream>

int main(){
  at::Tensor t = at::CUDA(at::kFloat).rand({128, 64, 2, 3, 3});
  std::cout << "Size of tensor t : " << t.size(1) << "\n";
  std::cout << "Stride of tensor t : " << t.stride(0) << "\n";
  //Tensor t = rand(CPU(kFloat).rand({10,3,1,4,4});i
  //Tensor t = rand(CPU(at::kFloat), {1,3});
  at::Tensor* p_t = &t;
  GridBase test(p_t,false);
}
