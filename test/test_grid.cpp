#include "grid.h"
#include "ATen/ATen.h"
#include <iostream>

int main(){
 at::Tensor A = at::CUDA(at::kFloat).rand({128, 10, 1, 3, 3});
 at::Tensor B = at::CUDA(at::kFloat).rand({128, 10, 1, 3, 3});

 for (int p=0; p < 1000000 ; p++){
   A = A + B;  
 }
 
}
