#include "bool_conversion.h"

namespace fluid {

bool toBool(const at::Tensor & self) {
   return self.equal(self.type().ones({}));
}

} // namespace fluid
