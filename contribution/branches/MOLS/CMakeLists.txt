######################################################################################
### 1) Project properties
######################################################################################

# set the project name
PROJECT(MOLS)

# check cmake version compatibility
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

INCLUDE(CMakeBackwardCompatibilityCXX)

# regular expression checking
INCLUDE_REGULAR_EXPRESSION("^.*$" "^$")

# set a language
ENABLE_LANGUAGE(CXX)
ENABLE_LANGUAGE(C)

INCLUDE(Dart OPTIONAL)

#####################################################################################

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

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    ADD_DEFINITIONS(-DCMAKE_VERBOSE_MAKEFILE=ON)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

IF(WIN32 AND NOT CYGWIN)
  IF(CMAKE_CXX_COMPILER MATCHES cl)
   IF(NOT WITH_SHARED_LIBS)
     IF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008")
       SET(CMAKE_CXX_FLAGS "/nologo /Gy")
       SET(CMAKE_CXX_FLAGS_DEBUG "/W3 /MTd /Z7 /Od")
       SET(CMAKE_CXX_FLAGS_RELEASE "/w /MT /O2 /wd4530")
       SET(CMAKE_CXX_FLAGS_MINSIZEREL "/MT /O2")
       SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MTd /Z7 /Od")
       SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")  
      ENDIF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008")
   ENDIF(NOT WITH_SHARED_LIBS)
  ENDIF(CMAKE_CXX_COMPILER MATCHES cl)  
ELSE(WIN32 AND NOT CYGWIN)
    IF(CMAKE_COMPILER_IS_GNUCXX)
       SET(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -fprofile-arcs -ftest-coverage -Wall -Wextra -Wno-unused-parameter")                       
       SET(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O2")
       SET(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -O6")       
    ENDIF(CMAKE_COMPILER_IS_GNUCXX)  
ENDIF(WIN32 AND NOT CYGWIN)


######################################################################################
### 2) Include the main configuration variables
######################################################################################

# Need the config file whose full path is given thanks to the "config" variable
INCLUDE("install.cmake")

######################################################################################


######################################################################################
### 3) Paths checking 
######################################################################################

IF(WIN32)
  SET (ABSOLUTE_PATH_REGEX "^[A-Z]:|^[a-z]:")
ELSE(WIN32)
  SET (ABSOLUTE_PATH_REGEX "^/")
ENDIF(WIN32)

SET(REQUIRED_PATHS "PARADISEO_EO_SRC_DIR" "PARADISEO_EO_BIN_DIR" "PARADISEO_MO_SRC_DIR" "PARADISEO_MO_BIN_DIR" "PARADISEO_MO_SRC_DIR" "PARADISEO_MOEO_BIN_DIR")
FOREACH (path ${REQUIRED_PATHS})
	  	IF(EXISTS ${${path}})
			MESSAGE (STATUS "Using ${path}=${${path}}")
		ELSE(EXISTS ${${path}})
			MESSAGE (FATAL_ERROR "\n Cannot find \"${${path}}\". Please, fill \"${config}\" with a correct value")
		ENDIF(EXISTS ${${path}})
		
		IF(NOT ${${path}} MATCHES "${ABSOLUTE_PATH_REGEX}")
			MESSAGE (FATAL_ERROR "${${path}} MUST BE an absolute path")
		ENDIF(NOT ${${path}} MATCHES "${ABSOLUTE_PATH_REGEX}")
ENDFOREACH (path ${REQUIRED_PATHS})
######################################################################################

######################################################################################
### 6) Where must cmake go now ?
######################################################################################

SUBDIRS(flowshop src doc test) 

######################################################################################

