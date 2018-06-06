#include <cmath>
#include <vector>
#include <sstream>
#include <omp.h>

#include "../test/load_manta_data.h"
#include "../test/plot_utils.h"

#include "ATen/ATen.h"

#include "fluid.h"

int main() {

  auto && Tfloat = CPU(at::kFloat);

  int dim = 3;
 
      std::string fn = std::to_string(dim) + "d_gravity.bin";
      at::Tensor undef1;
      at::Tensor U;
      at::Tensor flags;
      bool is3D;
      loadMantaBatch(fn, undef1, U, flags, undef1, is3D);
      assertNotAllEqual(U);
      assertNotAllEqual(flags);

      AT_ASSERT(is3D == (dim == 3), "Failed assert is3D");
      fn = std::to_string(dim) + "d_correctVelocity.bin";
      at::Tensor undef2;
      at::Tensor pressure;
      at::Tensor UManta;
      at::Tensor flagsManta;
      loadMantaBatch(fn, pressure, UManta, flagsManta, undef2, is3D);
      AT_ASSERT(is3D == (dim == 3), "Failed assert is3D");

      AT_ASSERT(flags.equal(flagsManta), "Flags changed!");
      AT_ASSERT(at::Scalar(at::max(at::abs(U - UManta))).toFloat() > 1e-5, "No velocities changed in Manta!");

  int b = flags.size(0);
  int d = flags.size(2);
  int h = flags.size(3);
  int w = flags.size(4);

  
}
