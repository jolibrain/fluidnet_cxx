SET(grid
     grid/grid.cpp
     grid/vec3.cpp
     grid/int3.h
     grid/stack_trace.h
     grid/grid.h
     grid/vec3.h
     grid/stack_trace.h)
ADD_LIBRARY(mygrid SHARED ${grid})
TARGET_INCLUDE_DIRECTORIES(mygrid PUBLIC
    "$<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR};${CMAKE_CURRENT_SOURCE_DIR}>/grid")
