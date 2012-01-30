######################################################################################
### CMake basic configuration
######################################################################################

# check cmake version compatibility
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# regular expression checking
INCLUDE_REGULAR_EXPRESSION("^.*$" "^$")

# set a language for the entire project.
ENABLE_LANGUAGE(CXX)
ENABLE_LANGUAGE(C)

####################################################################################


#####################################################################################
### Include required modules & utilities
#####################################################################################
INCLUDE(CMakeBackwardCompatibilityCXX)

INCLUDE(FindDoxygen)

INCLUDE(FindGnuplot)

INCLUDE(CheckLibraryExists)

INCLUDE(Dart OPTIONAL)    

INCLUDE(CPack)     

######################################################################################

       
#####################################################################################
### Manage the build type
#####################################################################################

# the user should choose the build type on windows environments,excepted under cygwin (default=none)
SET(CMAKE_DEFAULT_BUILD_TYPE "Release" CACHE STRING "Variable that stores the default CMake build type" FORCE)

FIND_PROGRAM(MEMORYCHECK_COMMAND
    NAMES purify valgrind
    PATHS
    "/usr/local/bin /usr/bin [HKEY_LOCAL_MACHINE\\SOFTWARE\\Rational Software\\Purify\\Setup;InstallFolder]"
    DOC "Path to the memory checking command, used for memory error detection.") 
       
IF(NOT CMAKE_BUILD_TYPE)
     SET( CMAKE_BUILD_TYPE 
          ${CMAKE_DEFAULT_BUILD_TYPE} CACHE STRING 
          "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." 
           FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)  
IF(CMAKE_COMPILER_IS_GNUCXX)
       SET(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -fprofile-arcs -ftest-coverage -Wall -Wextra -Wno-unused-parameter -Wno-ignored-qualifiers")                       
       SET(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O2")
       SET(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -O6")       
ENDIF(CMAKE_COMPILER_IS_GNUCXX)  

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    ADD_DEFINITIONS(-DCMAKE_VERBOSE_MAKEFILE=ON)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

#####################################################################################

######################################################################################
### compilation of examples?
######################################################################################

SET(ENABLE_CMAKE_EXAMPLE TRUE CACHE BOOL "Enable copy of benchs and parameters file?")

######################################################################################
### Test config
######################################################################################

IF (ENABLE_CMAKE_TESTING OR ENABLE_MINIMAL_CMAKE_TESTING)  
        ENABLE_TESTING()
ENDIF (ENABLE_CMAKE_TESTING OR ENABLE_MINIMAL_CMAKE_TESTING)
######################################################################################

#######################################################################################
### Paths to EO, MO and MOEO must be specified above.
#######################################################################################

SET(EO_SRC_DIR "${CMAKE_SOURCE_DIR}/../paradiseo-eo" CACHE PATH "ParadisEO-EO source directory" FORCE)
SET(EO_BIN_DIR "${CMAKE_BINARY_DIR}/../../paradiseo-eo/build" CACHE PATH "ParadisEO-EO binary directory" FORCE)
    
SET(MO_SRC_DIR "${CMAKE_SOURCE_DIR}/../paradiseo-mo" CACHE PATH "ParadisMO-MO source directory" FORCE)
SET(MO_BIN_DIR "${CMAKE_BINARY_DIR}/../../paradiseo-mo/build" CACHE PATH "ParadisMO-MO binary directory" FORCE)

SET(GPU_SRC_DIR "${CMAKE_SOURCE_DIR}/../paradiseo-gpu" CACHE PATH "ParadisEO-GPU source directory" FORCE)
SET(GPU_BIN_DIR "${CMAKE_BINARY_DIR}/../../paradiseo-gpu/build" CACHE PATH "ParadisEO-GPU binary directory" FORCE)

SET(PROBLEMS_SRC_DIR "${CMAKE_SOURCE_DIR}/../problems" CACHE PATH "Problems dependant source directory" FORCE)
    
######################################################################################
######################################################################################
### Subdirectories that CMake should process for EO, MO and GPU
######################################################################################

ADD_SUBDIRECTORY(doc)
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(tutorial)
######################################################################################
    
