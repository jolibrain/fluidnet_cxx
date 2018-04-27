#include "grid.h"

void GetPixelCenter(const vec3& pos, int32_t* ix,
                    int32_t* iy, int32_t* iz);

bool IsOutOfDomainReal(const vec3& pos, const FlagGrid& flags);

bool IsBlockedCell(const FlagGrid& flags,
                   int32_t i, int32_t j, int32_t k, int32_t b);

void ClampToDomain(const FlagGrid& flags,
                   int32_t* ix, int32_t* iy, int32_t* iz);

void ClampToDomainReal(vec3& pos, const FlagGrid& flags);

bool IsBlockedCellReal(const FlagGrid& flags,
                       const vec3& pos, int32_t b);

bool HitBoundingBox(const float* minB, const float* maxB, 
                    const float* origin, const float* dir,
                    float* coord);

bool calcRayBoxIntersection(const vec3& pos,
                            const vec3& dt,
                            const vec3& ctr,
                            const float hit_margin, vec3* ipos);

bool calcRayBorderIntersection(const vec3& pos,
                               const vec3& next_pos,
                               const FlagGrid& flags,
                               const float hit_margin,
                               vec3* ipos);

bool calcLineTrace(const vec3& pos, const vec3& delta,
                   const FlagGrid& flags, const int32_t ibatch,
                   vec3* new_pos, const bool do_line_trace);


