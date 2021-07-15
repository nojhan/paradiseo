
######################################################################################
# Project settings
######################################################################################

cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project("paradiseo-irace")

enable_language(CXX) # C++
set(CMAKE_CXX_STANDARD 17)

## Current version
set(VERSION_MAJOR 0 CACHE STRING "Major version number" )
set(VERSION_MINOR 1 CACHE STRING "Minor version number" )
set(VERSION_PATCH 0 CACHE STRING "Patch version number" )
mark_as_advanced(VERSION_MAJOR VERSION_MINOR VERSION_PATCH)


######################################################################################
# Configurable user settings
######################################################################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")

# put binaries in the build directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

# Dump used compiler flags.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


######################################################################################
# Dependencies
######################################################################################

# ParadisEO
set(PARADISEO_ROOT "../../../../paradiseo" CACHE PATH "Where to find ParadisEO")
set(PARADISEO_BUILD "${PARADISEO_ROOT}/build" CACHE PATH "Build dir of ParadisEO")

include_directories(${PARADISEO_ROOT})
include_directories(${PARADISEO_ROOT}/eo/src)
link_directories(${PARADISEO_BUILD}/lib)
set(PARADISEO_LIBRARIES ga eoutils eo)

# IOH
set(IOH_ROOT "~/code/IOHexperimenter/" CACHE PATH "Where to find IOHexperimenter")
find_path(IOH_HPP "ioh.hpp" PATHS ${IOH_ROOT}/include/)
# find_library(IOH_LIBRARY "IOH" PATHS ${IOH_ROOT} PATH_SUFFIXES release Release debug Debug build Build)

if(EXISTS ${IOH_HPP}) # AND EXISTS ${IOH_LIBRARY})
    message(STATUS "Found IOH in ${IOH_ROOT}")
    include_directories(${IOH_ROOT}/include/)
    include_directories(${IOH_ROOT}/external/fmt/include/)
    include_directories(${IOH_ROOT}/external/clutchlog/)
    link_directories(${IOH_ROOT}/release/external/fmt/)

else()
    if(NOT EXISTS ${IOH_HPP})
        message(FATAL_ERROR "Could not find `ioh.hpp` in: ${IOH_ROOT}/include/")
    endif()
    # if(NOT EXISTS ${IOH_LIBRARIES})
    #     message(FATAL_ERROR "Could not find `libIOH` in: ${IOH_ROOT}/[release|debug|build] (did you forget to compile it?)")
    # endif()
endif()


######################################################################################
# Start building
######################################################################################

add_executable(fastga fastga.cpp)
# target_link_libraries(fastga ${PARADISEO_LIBRARIES} ${IOH_LIBRARY} stdc++fs)
target_link_libraries(fastga ${PARADISEO_LIBRARIES} fmt)

