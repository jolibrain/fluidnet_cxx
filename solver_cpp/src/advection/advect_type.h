#pragma once

#include <string>

typedef enum {
  ADVECT_EULER_FLUIDNET = 0,
  ADVECT_MACCORMACK_FLUIDNET = 1,
} AdvectMethod;

AdvectMethod StringToAdvectMethod(const std::string& str);
