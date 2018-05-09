#pragma once

#include "ATen/ATen.h"

bool toBool(const at::Tensor & self){
   return self.equal(self.type().ones({}));
}
