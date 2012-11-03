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
    message(FATAL_ERROR "No compiler found !")
endif()

## Versioning
set(VERSION_MAJOR 2)
set(VERSION_MINOR 0)
set(VERSION_PATCH 1)

######################################################################################
### 2) Check dependencies
######################################################################################
 
## Optional
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

######################################################################################
### 4) Define and add module paths : EO, MO, MOEO
######################################################################################

## Paths
set(EO_SRC_DIR "${CMAKE_SOURCE_DIR}/eo" CACHE PATH "ParadisEO-EO source directory" FORCE)
set(EO_BIN_DIR "${CMAKE_BINARY_DIR}/eo" CACHE PATH "ParadisEO-EO binary directory" FORCE)
    
set(MO_SRC_DIR "${CMAKE_SOURCE_DIR}/mo" CACHE PATH "ParadisEO-MO source directory" FORCE)
set(MO_BIN_DIR "${CMAKE_BINARY_DIR}/mo" CACHE PATH "ParadisEO-MO binary directory" FORCE)

set(MOEO_SRC_DIR "${CMAKE_SOURCE_DIR}/moeo" CACHE PATH "ParadisEO-MOEO source directory" FORCE)
set(MOEO_BIN_DIR "${CMAKE_BINARY_DIR}/moeo" CACHE PATH "ParadisEO-MOEO binary directory" FORCE)

set(SMP_SRC_DIR "${CMAKE_SOURCE_DIR}/smp" CACHE PATH "ParadisEO-SMP source directory" FORCE)
set(SMP_BIN_DIR "${CMAKE_BINARY_DIR}/smp" CACHE PATH "ParadisEO-SMP binary directory" FORCE)

set(PEO_SRC_DIR "${CMAKE_SOURCE_DIR}/peo" CACHE PATH "ParadisEO-PEO source directory" FORCE)
set(PEO_BIN_DIR "${CMAKE_BINARY_DIR}/peo" CACHE PATH "ParadisEO-PEO binary directory" FORCE)

set(PROBLEMS_SRC_DIR "${CMAKE_SOURCE_DIR}/problems" CACHE PATH "Problems dependant source directory" FORCE)

set(CMAKE_BASE_SOURCE_DIR ${CMAKE_SOURCE_DIR})

## EO Module
set(EO_MODULE_NAME "Evolving Object")
set(CMAKE_SOURCE_DIR ${EO_SRC_DIR})
add_subdirectory(${CMAKE_SOURCE_DIR})

if(NOT EO_ONLY)
    ## MO Module
    set(MO_MODULE_NAME "ParadisEO-MO")
    set(CMAKE_SOURCE_DIR ${MO_SRC_DIR})
    add_subdirectory(${MO_SRC_DIR})

    ## MOEO Module
    set(MOEO_MODULE_NAME "ParadisEO-MOEO")
    set(CMAKE_SOURCE_DIR ${MOEO_SRC_DIR})
    add_subdirectory(${MOEO_SRC_DIR})
    
    ## SMP Module
    if(SMP)
        set(SMP_MODULE_NAME "ParadisEO-SMP")
        set(CMAKE_SOURCE_DIR ${SMP_SRC_DIR})
        add_subdirectory(${SMP_SRC_DIR})
    endif()
    
    ## PEO Module
    if(PEO)
        set(PEO_MODULE_NAME "ParadisEO-PEO")
        set(CMAKE_SOURCE_DIR ${PEO_SRC_DIR})
        add_subdirectory(${PEO_SRC_DIR})
    endif()
endif()

######################################################################################
### 5) Packaging : only in release !
######################################################################################

if("${CMAKE_BUILD_TYPE}" STREQUAL "Release")
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/Package.cmake)
endif()
