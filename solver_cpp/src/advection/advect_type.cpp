#include <sstream>
#include "ATen/ATen.h"
#include "advect_type.h"

AdvectMethod StringToAdvectMethod(const std::string& str) {
  if (str == "eulerFluidNet") {
    return ADVECT_EULER_FLUIDNET;
  } else if (str == "maccormackFluidNet") {
    return ADVECT_MACCORMACK_FLUIDNET;
  } else {
    std::stringstream ss;
    ss << "advection method (" << str << ") not supported (options "
       << "are: eulerFluidNet, maccormackFluidNet)";
    AT_ERROR("Error: Advection method not supported");
  }
}

