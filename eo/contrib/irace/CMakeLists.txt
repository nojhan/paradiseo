
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
find_path(IOH_PROBLEM_H "IOHprofiler_problem.h" PATHS ${IOH_ROOT}/src/Template/)
find_library(IOH_LIBRARY "IOH" PATHS ${IOH_ROOT} PATH_SUFFIXES release Release debug Debug build Build)

if(EXISTS ${IOH_PROBLEM_H} AND EXISTS ${IOH_LIBRARY})
    message(STATUS "Found IOH in ${IOH_ROOT}")
    include_directories(${IOH_ROOT}/build/Cpp/src/)
    link_directories(${IOH_ROOT}/build/Cpp/bin/)

    # Workaround IOH's poorly designed headers inclusion scheme.
    SET(PROBLEMS_BBOB_DIR "src/Problems/BBOB")
    SET(PROBLEMS_BBOB_COMMON_DIR "src/Problems/BBOB/bbob_common_used_functions")
    SET(PROBLEMS_COMMON_DIR "src/Problems/common_used_functions")
    SET(PROBLEMS_PBO_DIR "src/Problems/PBO")
    SET(PROBLEMS_WMODEL_DIR "src/Problems/WModel")
    SET(PROBLEMS_PYTHON_DIR "src/Problems/Python")
    SET(SUITES_DIR "src/Suites")
    SET(TEMPLATE_DIR "src/Template")
    SET(TEMPLATE_EXPERIMENTS_DIR "src/Template/Experiments")
    SET(TEMPLATE_LOGGERS_DIR "src/Template/Loggers")
    SET(IOHEXPERIMENTER_DIR
      "${IOH_ROOT}/${PROBLEMS_COMMON_DIR}"
      "${IOH_ROOT}/${PROBLEMS_BBOB_DIR}"
      "${IOH_ROOT}/${PROBLEMS_BBOB_COMMON_DIR}"
      "${IOH_ROOT}/${PROBLEMS_PBO_DIR}"
      "${IOH_ROOT}/${PROBLEMS_WMODEL_DIR}"
      "${IOH_ROOT}/${PROBLEMS_PYTHON_DIR}"
      "${IOH_ROOT}/${SUITES_DIR}"
      "${IOH_ROOT}/${TEMPLATE_DIR}"
      "${IOH_ROOT}/${TEMPLATE_EXPERIMENTS_DIR}"
      "${IOH_ROOT}/${TEMPLATE_LOGGERS_DIR}"
    )
    include_directories(${IOHEXPERIMENTER_DIR})

else()
    if(NOT EXISTS ${IOH_PROBLEM_H})
        message(FATAL_ERROR "Could not find `IOHprofiler_problem.h` in: ${IOH_ROOT}/src/Template/ (did you forget to compile it?)")
    endif()
    if(NOT EXISTS ${IOH_LIBRARIES})
        message(FATAL_ERROR "Could not find `libIOH` in: ${IOH_ROOT}/[release|debug|build] (did you forget to compile it?)")
    endif()
endif()


######################################################################################
# Start building
######################################################################################

add_executable(fastga fastga.cpp)
target_link_libraries(fastga ${PARADISEO_LIBRARIES} ${IOH_LIBRARY})

