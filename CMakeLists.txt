# ParadiseO

######################################################################################
### 0) Check the CMake version
######################################################################################

cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

######################################################################################
### 1) Define the project
######################################################################################

## Name
project("ParadisEO")

## Language
enable_language(CXX)

## Test the presence of a compiler
if("${CMAKE_CXX_COMPILER}" STREQUAL "" OR "${CMAKE_C_COMPILER}" STREQUAL "")
    message(FATAL_ERROR "No compiler found!")
endif()

## Versioning
set(VERSION_MAJOR 2 CACHE STRING "Major version number" )
set(VERSION_MINOR 1 CACHE STRING "Minor version number" )
set(VERSION_PATCH 0 CACHE STRING "Patch version number" )

SET(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}" CACHE INTERNAL "Package version" FORCE)
SET(VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}${PROJECT_VERSION_MISC}" CACHE INTERNAL "Global version" FORCE)
SET(GLOBAL_VERSION "${VERSION}")

######################################################################################
### 2) Check dependencies
######################################################################################
 
## Optional
set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake/module" CACHE INTERNAL "Cmake module" FORCE)
include(FindDoxygen OPTIONAL)

######################################################################################
### 3) Include CMake files
######################################################################################

## Configuration file for building type and flags
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake)

## Macro file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Macro.cmake)

## Custom Targets file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Target.cmake)
 
set(CMAKE_BASE_SOURCE_DIR ${CMAKE_SOURCE_DIR})

# All libraries are built in <build_dir>/lib/
set(      EO_BIN_DIR "${CMAKE_BINARY_DIR}" CACHE INTERNAL "ParadisEO-EO       binary directory" FORCE)
set(     EDO_BIN_DIR "${CMAKE_BINARY_DIR}" CACHE INTERNAL "ParadisEO-EDO      binary directory" FORCE)
set(      MO_BIN_DIR "${CMAKE_BINARY_DIR}" CACHE INTERNAL "ParadisEO-MO       binary directory" FORCE)
set(    MOEO_BIN_DIR "${CMAKE_BINARY_DIR}" CACHE INTERNAL "ParadisEO-MOEO     binary directory" FORCE)
set(     SMP_BIN_DIR "${CMAKE_BINARY_DIR}" CACHE INTERNAL "ParadisEO-SMP      binary directory" FORCE)
set(   EOMPI_BIN_DIR "${CMAKE_BINARY_DIR}" CACHE INTERNAL "ParadisEO-EOMPI    binary directory" FORCE)
set(EOSERIAL_BIN_DIR "${CMAKE_BINARY_DIR}" CACHE INTERNAL "ParadisEO-EOSERIAL binary directory" FORCE)

######################################################################################
### 4) Choose module(s) 
######################################################################################

set(EO_ONLY  "false" CACHE BOOL "Only build EO and not the other modules")
set(MO       "false" CACHE BOOL "Build the MO module")
set(EDO      "false" CACHE BOOL "Build the EDO module")
set(MOEO     "false" CACHE BOOL "Build the MOEO module")
set(SMP      "false" CACHE BOOL "Build the SMP module")
set(EOMPI    "false" CACHE BOOL "Build the EOMPI module")
set(EOSERIAL "false" CACHE BOOL "Build the EOSERIAL module")

set(ENABLE_OPENMP  "false" CACHE BOOL "Build EO with the OpenMP support (shared-memory parallel evaluators on multi-core)")
set(ENABLE_GNUPLOT "false" CACHE BOOL "Build EO with the GNUplot support (real-time convergence plotting)")
set(EDO_USE_LIB    "Eigen3" CACHE STRING "Which linear algebra library to use to build EDO ('UBlas' or 'Eigen3', Eigen3 is recommended)")

set(      EO_TEST_DIR "${CMAKE_SOURCE_DIR}/test/eo"         CACHE INTERNAL "ParadisEO-EO       test directory" FORCE)
set(     EDO_TEST_DIR "${CMAKE_SOURCE_DIR}/test/edo"        CACHE INTERNAL "ParadisEO-EDO      test directory" FORCE)
set(      MO_TEST_DIR "${CMAKE_SOURCE_DIR}/test/mo"         CACHE INTERNAL "ParadisEO-MO       test directory" FORCE)
set(    MOEO_TEST_DIR "${CMAKE_SOURCE_DIR}/test/moeo"       CACHE INTERNAL "ParadisEO-MOEO     test directory" FORCE)
set(     SMP_TEST_DIR "${CMAKE_SOURCE_DIR}/test/smp"        CACHE INTERNAL "ParadisEO-SMP      test directory" FORCE)
set(   EOMPI_TEST_DIR "${CMAKE_SOURCE_DIR}/test/eompi"      CACHE INTERNAL "ParadisEO-EOMPI    test directory" FORCE)
set(EOSERIAL_TEST_DIR "${CMAKE_SOURCE_DIR}/test/eoserial"   CACHE INTERNAL "ParadisEO-EOSERIAL test directory" FORCE)

set(      EO_TUTORIAL_DIR "${CMAKE_SOURCE_DIR}/tutorial/eo"         CACHE INTERNAL "ParadisEO-EO       tutorial directory" FORCE)
set(     EDO_TUTORIAL_DIR "${CMAKE_SOURCE_DIR}/tutorial/edo"        CACHE INTERNAL "ParadisEO-EDO      tutorial directory" FORCE)
set(      MO_TUTORIAL_DIR "${CMAKE_SOURCE_DIR}/tutorial/mo"         CACHE INTERNAL "ParadisEO-MO       tutorial directory" FORCE)
set(    MOEO_TUTORIAL_DIR "${CMAKE_SOURCE_DIR}/tutorial/moeo"       CACHE INTERNAL "ParadisEO-MOEO     tutorial directory" FORCE)
set(     SMP_TUTORIAL_DIR "${CMAKE_SOURCE_DIR}/tutorial/smp"        CACHE INTERNAL "ParadisEO-SMP      tutorial directory" FORCE)
set(   EOMPI_TUTORIAL_DIR "${CMAKE_SOURCE_DIR}/tutorial/eompi"      CACHE INTERNAL "ParadisEO-EOMPI    tutorial directory" FORCE)
set(EOSERIAL_TUTORIAL_DIR "${CMAKE_SOURCE_DIR}/tutorial/eoserial"   CACHE INTERNAL "ParadisEO-EOSERIAL tutorial directory" FORCE)

######################################################################################
### 5) Add sources
######################################################################################

add_subdirectory(src)
add_subdirectory(doc)

if(ENABLE_CMAKE_EXAMPLE)
    add_subdirectory(tutorial)
endif(ENABLE_CMAKE_EXAMPLE)

if(ENABLE_CMAKE_TESTING)
    add_subdirectory(test)
endif(ENABLE_CMAKE_TESTING)


######################################################################################
### 6) Packaging : only in release !
######################################################################################

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Package.cmake)
endif()
