SET(GRID
     grid/grid_scalar.cpp
    )
#SET(ADVECTION
#     advection/advection.cpp
#     advection/advect_type.cpp
#     advection/calc_line_trace.cpp
#   ) 

#ADD_LIBRARY(FluidNet SHARED ${GRID} ${ADVECTION})
#TARGET_INCLUDE_DIRECTORIES(FluidNet
#   PUBLIC
#       $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/grid;
#       ${CMAKE_CURRENT_SOURCE_DIR}/advection;
#       ${CMAKE_CURRENT_SOURCE_DIR}>
#)
ADD_LIBRARY(FluidNet SHARED ${GRID})
TARGET_INCLUDE_DIRECTORIES(FluidNet
   PUBLIC
       $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/grid;
       ${CMAKE_CURRENT_SOURCE_DIR}>
)

