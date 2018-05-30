#pragma once

#include "calc_line_trace.h"
#include "advect_type.h"
#include "grid/grid.h"

namespace fluid {
// ****************************************************************************
// Advect Scalar
// ****************************************************************************

// Euler advection with line trace (as in Fluid Net)
T SemiLagrangeEulerFluidNet
(
    FlagGrid& flags,
    MACGrid& vel,
    RealGrid& src,
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
T SemiLagrangeEulerFluidNetSavePos
(
    FlagGrid& flags,
    MACGrid& vel,
    RealGrid& src,
    float dt,
    int order_space,
    int32_t i, int32_t j, int32_t k, int32_t b,
    const bool line_trace,
    const bool sample_outside_fluid,
    VecGrid& pos
);

T MacCormackCorrect
(
    FlagGrid& flags,
    const RealGrid& old,
    const RealGrid& fwd,
    const RealGrid& bwd,
    const float strength,
    bool is_levelset,
    int32_t i, int32_t j, int32_t k, int32_t b
);

void getMinMax(T& minv, T& maxv, const T& val) {
  if (toBool(val < minv)) {
    minv = val;
  }
  if (toBool(val > maxv)) {
    maxv = val;
  }
};

// FluidNet clamp routine. It is a search around a single input
// position for min and max values. If no valid values are found, then
// false is returned (indicating that a clamp shouldn't be performed) otherwise
// true is returned (and the clamp min and max bounds are set).
bool getClampBounds
(
    RealGrid src,
    T pos,
    const int32_t b,
    FlagGrid flags,
    const bool sample_outside_fluid,
    T* clamp_min,
    T* clamp_max
) ;

T MacCormackClampFluidNet
(
    FlagGrid& flags,
    MACGrid& vel,
    const RealGrid& dst,
    const RealGrid& src,
    const RealGrid& fwd,
    float dt,
    const VecGrid& fwd_pos,
    const VecGrid& bwd_pos,
    const bool sample_outside_fluid,
    int32_t i, int32_t j, int32_t k, int32_t b
);

// Advect scalar field 'p' by the input vel field 'u'.
// 
// @input dt - timestep (seconds).
// @input s - input scalar field to advect
// @input U - input vel field (size(2) can be 2 or 3, indicating 2D / 3D)
// @input flags - input occupancy grid
// @input method - OPTIONAL - "eulerFluidNet", "maccormackFluidNet"
// @input inplace - Perform inplace advection or return sDst
// @input sDst - If inplace is false this will be the returned scalar field. Otherwise advection will be performed in-place.
// @param sampleOutsideFluid - OPTIONAL - For density advection we do not want
// to advect values inside non-fluid cells and so this should be set to false.
// For other quantities (like temperature), this should be true.
// @param maccormackStrength - OPTIONAL - (default 0.75) A strength parameter
// will make the advection eularian (with values interpolating in between). A
// value of 1 (which implements the update from An Unconditionally Stable
// MaCormack Method) tends to add too much high-frequency detail
// @param boundaryWidth - OPTIONAL - boundary width. (default 1)

void advectScalar
(
    float dt,
    at::Tensor& tensor_flags,
    at::Tensor& tensor_u,
    at::Tensor& tensor_s,
    const bool inplace, 
    at::Tensor& tensor_s_dst,
    const std::string method_str = "maccormackFluidNet",
    const int32_t boundary_width = 1,
    const bool sample_outside_fluid = false,
    const float maccormack_strength = 0.75
);

// ****************************************************************************
// Advect Velocity
// ***************************************************************************

T SemiLagrangeEulerFluidNetMAC(
    FlagGrid& flags,
    MACGrid& vel,
    MACGrid& src,
    float dt,
    int order_space,
    const bool line_trace,
    int32_t i, int32_t j, int32_t k, int32_t b);

T SemiLagrangeMAC(
    FlagGrid& flags,
    MACGrid& vel,
    MACGrid& src,
    float dt,
    int order_space,
    int32_t i, int32_t j, int32_t k, int32_t b);

T MacCormackCorrectMAC(
    FlagGrid& flags,
    const MACGrid& old,
    const MACGrid& fwd,
    const MACGrid& bwd,
    const float strength,
    int32_t i, int32_t j, int32_t k, int32_t b) ;

template <int32_t c>
T doClampComponentMAC(
    const T& gridSize,
    T dst,
    const MACGrid& orig,
    T fwd,
    const T& pos, const T& vel,
    int32_t b);

T MacCormackClampMAC(
    FlagGrid& flags,
    MACGrid& vel,
    T dval,
    const MACGrid& orig,
    const MACGrid& fwd,
    float dt,
    int32_t i, int32_t j, int32_t k, int32_t b);

// Advect velocity field 'u' by itself and store in uDst.
// 
// @input dt - timestep (seconds).
// @input U - input vel field (size(2) can be 2 or 3, indicating 2D / 3D)
// @input flags - input occupancy grid
// @input method - OPTIONAL - "eulerFluidNet", "maccormackFluidNet" (default)
// @input inplace - If true, performs advection in place.
// @input UDst - If inplace is true then this will be the returned
// velocity field. Otherwise advection will be performed in-place.
// @input maccormackStrength - OPTIONAL - (default 0.75) A strength parameter
// will make the advection more 1st order (with values interpolating in
// between). A value of 1 (which implements the update from "An Unconditionally
// Stable MaCormack Method") tends to add too much high-frequency detail.
// @input boundaryWidth - OPTIONAL - boundary width. (default 1)

void advectVel
(
    float dt,
    T& tensor_flags,
    T& tensor_u,
    const bool inplace,
    T& tensor_u_dst,
    T& tensor_fwd,
    T& tensor_bwd,
    const std::string method_str = "maccormackFluidNet",
    const int32_t boundary_width = 1,
    const float maccormack_strength = 0.75
);

} // namespace fluid
