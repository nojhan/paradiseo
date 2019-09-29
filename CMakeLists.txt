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
if(NOT DOXYGEN_FOUND)
    message(WARNING "Doxygen was not found, install it if you want to generate the documentation.")
else()
    message(STATUS "Doxygen found, use the target `doc` to generate the documentation.")
endif()

######################################################################################
### 3) Include CMake files
######################################################################################

## Configuration file for building type and flags
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake)

## Macro file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Macro.cmake)

## Custom Targets file
include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Target.cmake)

######################################################################################
### 4) Define and add module paths : EO, MO, MOEO
######################################################################################

## Paths to sources of modules
set(  EO_SRC_DIR "${CMAKE_SOURCE_DIR}/eo"   CACHE INTERNAL "ParadisEO-EO   source directory" FORCE)
set( EDO_SRC_DIR "${CMAKE_SOURCE_DIR}/edo"  CACHE INTERNAL "ParadisEO-EDO  source directory" FORCE)
set(  MO_SRC_DIR "${CMAKE_SOURCE_DIR}/mo"   CACHE INTERNAL "ParadisEO-MO   source directory" FORCE)
set(MOEO_SRC_DIR "${CMAKE_SOURCE_DIR}/moeo" CACHE INTERNAL "ParadisEO-MOEO source directory" FORCE)
set( SMP_SRC_DIR "${CMAKE_SOURCE_DIR}/smp"  CACHE INTERNAL "ParadisEO-SMP  source directory" FORCE)
set( MPI_SRC_DIR "${CMAKE_SOURCE_DIR}/eo/src/mpi"  CACHE INTERNAL "ParadisEO-MPI source directory" FORCE)

set(PROBLEMS_SRC_DIR "${CMAKE_SOURCE_DIR}/problems" CACHE INTERNAL "Problems dependant source directory" FORCE)

set(CMAKE_BASE_SOURCE_DIR ${CMAKE_SOURCE_DIR})

# All libraries are built in <build_dir>/lib/
set(  EO_BIN_DIR "${CMAKE_BINARY_DIR}" CACHE INTERNAL "ParadisEO-EO   binary directory" FORCE)
set( EDO_BIN_DIR "${CMAKE_BINARY_DIR}" CACHE INTERNAL "ParadisEO-EDO  binary directory" FORCE)
set(  MO_BIN_DIR "${CMAKE_BINARY_DIR}" CACHE INTERNAL "ParadisEO-MO   binary directory" FORCE)
set(MOEO_BIN_DIR "${CMAKE_BINARY_DIR}" CACHE INTERNAL "ParadisEO-MOEO binary directory" FORCE)
set( SMP_BIN_DIR "${CMAKE_BINARY_DIR}" CACHE INTERNAL "ParadisEO-SMP  binary directory" FORCE)
set( MPI_BIN_DIR "${CMAKE_BINARY_DIR}" CACHE INTERNAL "ParadisEO-MPI  binary directory" FORCE)


set(EO_ONLY "false" CACHE BOOL "Only build EO and not the other modules")
set(ENABLE_OPENMP "false" CACHE BOOL "Build EO with the OpenMP support (shared-memory parallel evaluators on multi-core)")
set(ENABLE_GNUPLOT "false" CACHE BOOL "Build EO with the GNUplot support (real-time convergence plotting)")
set(EDO "false" CACHE BOOL "Build the EDO module")
set(EDO_USE_LIB "Eigen3" CACHE STRING "Which linear algebra library to use to build EDO ('UBlas' or 'Eigen3', Eigen3 is recommended)")
set(SMP "false" CACHE BOOL "Build the SMP module")
set(MPI "false" CACHE BOOL "Build the MPI module")

## EO Module
set(EO_MODULE_NAME "Evolving Object")
set(CMAKE_SOURCE_DIR ${EO_SRC_DIR})
add_subdirectory(${EO_SRC_DIR})

if(NOT EO_ONLY)
    ## MO Module
    set(MO_MODULE_NAME "Moving objects")
    set(CMAKE_SOURCE_DIR ${MO_SRC_DIR})
    add_subdirectory(${MO_SRC_DIR})

    ## EDO Module
    if(EDO)
        set(EDO_MODULE_NAME "Evolving Distribution Objects")
        set(CMAKE_SOURCE_DIR ${EDO_SRC_DIR})
        add_subdirectory(${EDO_SRC_DIR})
    endif()

    ## MOEO Module
    set(MOEO_MODULE_NAME "Multi-Objectives EO")
    set(CMAKE_SOURCE_DIR ${MOEO_SRC_DIR})
    add_subdirectory(${MOEO_SRC_DIR})

    ## SMP Module
    if(SMP)
        set(SMP_MODULE_NAME "Symmetric Multi-Processing")
        set(CMAKE_SOURCE_DIR ${SMP_SRC_DIR})
        add_subdirectory(${SMP_SRC_DIR})
    endif()

    ## MPI Module
    if(MPI)
        find_package(MPI REQUIRED)
        add_definitions(-DWITH_MPI)
        set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
        set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})
        include_directories(${MPI_INCLUDE_PATH})
        add_subdirectory(${MPI_SRC_DIR})
    endif()
endif()

######################################################################################
### 5) Packaging : only in release !
######################################################################################

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Package.cmake)
endif()


# Add all targets to the build-tree export set
export(TARGETS eo FILE "${PROJECT_BINARY_DIR}/paradiseo-config.cmake")

# Export the package for use from the build-tree
# (this registers the build-tree with a global CMake-registry)
export(PACKAGE paradiseo)

