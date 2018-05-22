SET(GRID
     grid/bool_conversion.cpp 
     grid/grid.cpp
    )
SET(ADVECTION
     advection/advection.cpp
     advection/bcs.cpp
     advection/div.cpp
     advection/solveLinearSys.cpp
     advection/updateVel.cpp
     advection/advect_type.cpp
     advection/calc_line_trace.cpp
   ) 

ADD_LIBRARY(FluidNet SHARED ${GRID} ${ADVECTION})
TARGET_INCLUDE_DIRECTORIES(FluidNet
   PUBLIC
       $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/grid;
       ${CMAKE_CURRENT_SOURCE_DIR}/advection;
       ${CMAKE_CURRENT_SOURCE_DIR}>
)

