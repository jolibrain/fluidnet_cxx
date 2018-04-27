#include "grid.h"
#include "ATen/ATen.h"
#include <sstream>

using namespace at;
// a simple sum kernel (for CPU only)
template<typename T>
struct sum_op {
  // dispatch handles variable arguments for you
  Tensor CPU(Tensor & x_)
  {
    Tensor x = x_.contiguous();
    auto x_p = x.data<T>();
    int64_t size = x.numel();
    T sum = 0;
    for(int64_t i = 0; i < size; i++) {
      sum += x_p[i];
    }
    return ::at::CPU(x_.type().scalarType()).scalarTensor(sum);
  };
  T CUDA(Tensor& x) {
    throw std::invalid_argument("device not supported");
  };
};

int main(){

Tensor a = CUDA(kFloat).rand({20, 3, 1, 5, 5});
MACGrid mac(&a, false);


   

  
}
