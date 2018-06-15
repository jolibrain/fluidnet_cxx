#include "ATen/ATen.h"

namespace fluid {

typedef at::Tensor T;

T interpol(T& self, T& pos);

void interpol1DWithFluid(
    const T& val_a, const T& is_fluid_a,
    const T& val_b, const T& is_fluid_b,
    const T& t_a, const T& t_b,
    T& is_fluid_ab, T& val_ab);

T interpolWithFluid(T& self, T& flags, T& pos);

T getCentered(const T& self);

T getAtMACX(const T& self);
T getAtMACY(const T& self);
T getAtMACZ(const T& self);

T interpolComponent(const T& self, const T& pos, int c);

T curl(const T& self);

} // namespace fluid
