// Copyright 2016 Google Inc, NYU.
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <assert.h>
#include <algorithm>
#include "ATen/ATen.h"

#include "stack_trace.cpp"
#include "cell_type.h"

// This type is common to both float and double implementations and so has
// to be defined outside tfluids.cc.
#include "int3.h"
#include "advect_type.h"

// Some common functions
inline int32_t clamp(const int32_t x, const int32_t low, const int32_t high) {
  return std::max<int32_t>(std::min<int32_t>(x, high), low);
}

// Expand the CPU types (float and double).  This actually instantiates the
// functions. Note: the order here is important.

#define tfluids_(NAME) tfluids_ ## Real ## NAME

#define real float
#define accreal double
#define Real Float
#define TH_REAL_IS_FLOAT
#include "vec3.cpp"
#include "grid.cpp"
#include "tfluids.cpp"
#undef accreal
#undef real
#undef Real
#undef THInf
#undef TH_REAL_IS_FLOAT

