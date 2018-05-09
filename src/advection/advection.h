#include "calc_line_trace.h"
#include "advect_type.h"
#include "../grid/grid.h"

// ****************************************************************************
// advectVel
// ****************************************************************************

// Euler advection with line trace (as in Fluid Net)
float SemiLagrangeEulerFluidNet
(
    FlagGrid& flags,
    MACGrid& vel,
    FloatGrid& src,
    float dt,
    int order_space,
    int32_t i, int32_t j, int32_t k, int32_t b,
    const bool line_trace,
    const bool sample_outside_fluid
);

// Same kernel as previous one, except that it saves the 
// particle trace position. This is used for the Fluid Net
// MacCormack routine (it does
// a local search around these positions in clamp routine).
float SemiLagrangeEulerFluidNetSavePos
(
    FlagGrid& flags,
    MACGrid& vel,
    FloatGrid& src,
    float dt,
    int order_space,
    int32_t i, int32_t j, int32_t k, int32_t b,
    const bool line_trace,
    const bool sample_outside_fluid,
    VecGrid& pos
);

float MacCormackCorrect
(
    FlagGrid& flags,
    const FloatGrid& old,
    const FloatGrid& fwd,
    const FloatGrid& bwd,
    const float strength,
    bool is_levelset,
    int32_t i, int32_t j, int32_t k, int32_t b
);

void getMinMax(float& minv, float& maxv, const float& val) {
  if (val < minv) {
    minv = val;
  }
  if (val > maxv) {
    maxv = val;
  }
};

float clamp(const float val, const float min, const float max) {
  return std::min<float>(max, std::max<float>(min, val));
};

// FluidNet clamp routine. It is a search around a single input
// position for min and max values. If no valid values are found, then
// false is returned (indicating that a clamp shouldn't be performed) otherwise
// true is returned (and the clamp min and max bounds are set).
static float getClampBounds
(
    FloatGrid src,
    vec3 pos,
    const int32_t b,
    FlagGrid flags,
    const bool sample_outside_fluid,
    float* clamp_min,
    float* clamp_max
) ;

float MacCormackClampFluidNet
(
    FlagGrid& flags,
    MACGrid& vel,
    const FloatGrid& dst,
    const FloatGrid& src,
    const FloatGrid& fwd,
    float dt,
    const VecGrid& fwd_pos,
    const VecGrid& bwd_pos,
    const bool sample_outside_fluid,
    int32_t i, int32_t j, int32_t k, int32_t b
);

// Main routine for scalar advection
void advectScalar
(
    float dt,
    at::Tensor* tensor_flags,
    at::Tensor* tensor_u,
    at::Tensor* tensor_s,
    at::Tensor* tensor_s_dst,
    at::Tensor* tensor_fwd,
    at::Tensor* tensor_bwd,
    at::Tensor* tensor_fwd_pos,
    at::Tensor* tensor_bwd_pos,
    const bool is_3d,
    const std::string method_str,
    const int32_t boundary_width = 1,
    const bool sample_outside_fluid = false,
    const float maccormack_strength = 1.
);

// ****************************************************************************
// advectVel
// ***************************************************************************

vec3 SemiLagrangeEulerFluidNetMAC(
    FlagGrid& flags,
    MACGrid& vel,
    MACGrid& src,
    float dt,
    int order_space,
    const bool line_trace,
    int32_t i, int32_t j, int32_t k, int32_t b);

vec3 SemiLagrangeMAC(
    FlagGrid& flags,
    MACGrid& vel,
    MACGrid& src,
    float dt,
    int order_space,
    int32_t i, int32_t j, int32_t k, int32_t b);

vec3 MacCormackCorrectMAC(
    FlagGrid& flags,
    const MACGrid& old,
    const MACGrid& fwd,
    const MACGrid& bwd,
    const float strength,
    int32_t i, int32_t j, int32_t k, int32_t b) ;

template <int32_t c>
float doClampComponentMAC(
    const Int3& gridSize,
    float dst,
    const MACGrid& orig,
    float fwd,
    const vec3& pos, const vec3& vel,
    int32_t b);

vec3 MacCormackClampMAC(
    FlagGrid& flags,
    MACGrid& vel,
    vec3 dval,
    const MACGrid& orig,
    const MACGrid& fwd,
    float dt,
    int32_t i, int32_t j, int32_t k, int32_t b);

void advectVel
(
    float dt,
    at::Tensor* tensor_flags,
    at::Tensor* tensor_u,
    at::Tensor* tensor_u_dst,
    at::Tensor* tensor_fwd,
    at::Tensor* tensor_bwd,
    const bool is_3d,
    const std::string method_str,
    const int32_t boundary_width,
    const float maccormack_strength
);



