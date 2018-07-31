CMAKE_MINIMUM_REQUIRED(VERSION 3.0)
PROJECT("fluidnet_cxx" LANGUAGES CXX)

INCLUDE(CMakeDependentOption)
OPTION(FLUID_TEST "Build Fluid test binaries" OFF)

SET(CMAKE_BUILD_TYPE Release)

# Include manually ATen
SET(ATEN_INCLUDE_DIR "/usr/local/lib/python3.5/dist-packages/torch/lib/include")
LINK_DIRECTORIES("/usr/local/lib/python3.5/dist-packages/torch/lib/")
INCLUDE_DIRECTORIES(${ATEN_INCLUDE_DIR})

SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -ggdb -std=c++11")

FIND_PACKAGE(OpenCV REQUIRED)
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS})

ADD_SUBDIRECTORY(src)
IF (FLUID_TEST)
  ADD_SUBDIRECTORY(test)
ELSE ()
  ADD_SUBDIRECTORY(simulate)
ENDIF()



