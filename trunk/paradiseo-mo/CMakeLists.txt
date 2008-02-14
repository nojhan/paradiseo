
######################################################################################
### 0) If you want to set your own variables in mo-conf.cmake and avoid the cmd line
######################################################################################

INCLUDE(mo-conf.cmake OPTIONAL)

######################################################################################


######################################################################################
### 1) Project properties
######################################################################################

# set the project name
PROJECT(ParadisEO-MO)

SET(PACKAGE_BUGREPORT "paradiseo-help@lists.gforge.inria.fr" CACHE STRING "Package bug report" FORCE)
SET(PACKAGE_NAME "ParadisEO-MO Moving Objects" CACHE STRING "Package name" FORCE)
SET(PACKAGE_STRING "ParadisEO-MO 1.0" CACHE STRING "Package string full name" FORCE)
SET(PACKAGE_VERSION "1.0" CACHE STRING "Package version" FORCE)
SET(GLOBAL_VERSION "1.0" CACHE STRING "Global version" FORCE)
SET(VERSION "1.0" CACHE STRING "Version" FORCE)

# check cmake version compatibility
CMAKE_MINIMUM_REQUIRED(VERSION 2.4 FATAL_ERROR)

# regular expression checking
INCLUDE_REGULAR_EXPRESSION("^.*$" "^$")

# set a language for the entire project.
ENABLE_LANGUAGE(CXX)
ENABLE_LANGUAGE(C)

#####################################################################################


#####################################################################################
### 2) Include required modules
#####################################################################################

INCLUDE(CMakeBackwardCompatibilityCXX)

INCLUDE(FindDoxygen)

INCLUDE(FindGnuplot)

# check for some functions
INCLUDE(CheckLibraryExists)

INCLUDE(Dart OPTIONNAL)         
                           
# the project can have a DartConfig.cmake file
INCLUDE(DartConfig.cmake OPTIONNAL)
######################################################################################


######################################################################################
### 3) Include the main configuration variables
######################################################################################

# The "config" variable must be provided on the command line
IF(NOT DEFINED config OR NOT config)
  MESSAGE(FATAL_ERROR  "The \"config\" variable must be set on the command line to 
  						give the path of the install configuration file. ")
ENDIF(NOT DEFINED config OR NOT config)

# Need the config file whose full path is given thanks to the "config" variable
INCLUDE(${config})

######################################################################################


######################################################################################
### 4) Paths checking 
######################################################################################

IF(WIN32)
  SET (ABSOLUTE_PATH_REGEX "^[A-Z]:|^[a-z]:")
ELSE(WIN32)
  SET (ABSOLUTE_PATH_REGEX "^/")
ENDIF(WIN32)

SET(REQUIRED_PATHS "EO_SRC_DIR" "EO_BIN_DIR")
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


#####################################################################################
### 5) Manage the build type
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
  
IF(WIN32 AND NOT CYGWIN)
  IF(CMAKE_CXX_COMPILER MATCHES cl)
   IF(NOT WITH_SHARED_LIBS)
     IF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005")
       SET(CMAKE_CXX_FLAGS "/nologo /W3 /Gy")
       SET(CMAKE_CXX_FLAGS_DEBUG "/MTd /Z7 /Od")
       SET(CMAKE_CXX_FLAGS_RELEASE "/MT /O2")
       SET(CMAKE_CXX_FLAGS_MINSIZEREL "/MT /O2")
       SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MTd /Z7 /Od")
       SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")  
     ENDIF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005")
   ENDIF(NOT WITH_SHARED_LIBS)
  ENDIF(CMAKE_CXX_COMPILER MATCHES cl)  
ELSE(WIN32 AND NOT CYGWIN)
    IF(CMAKE_COMPILER_IS_GNUCXX)
       SET(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g -Wall -pedantic -Wextra -Wno-import -Winit-self  -Wmissing-include-dirs -Wunused-parameter -Wundef  -Wshadow  -Wunsafe-loop-optimizations -Wcast-align -Wconversion -Wsign-compare -Winline -Wdisabled-optimization")                       
       SET(CMAKE_CXX_FLAGS_RELEASE  "${CMAKE_CXX_FLAGS_RELEASE} -O2")
       SET(CMAKE_CXX_FLAGS_MINSIZEREL "${CMAKE_CXX_FLAGS_MINSIZEREL} -O6")       
    ENDIF(CMAKE_COMPILER_IS_GNUCXX)  
ENDIF(WIN32 AND NOT CYGWIN)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    ADD_DEFINITIONS(-DCMAKE_VERBOSE_MAKEFILE=ON)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

#####################################################################################


######################################################################################
### 6) Where must cmake go now ?
######################################################################################

SUBDIRS(doc test tutorial)

######################################################################################


######################################################################################
### 7) Test config
######################################################################################

#SET(ENABLE_CMAKE_TESTING TRUE CACHE BOOL "Should we test using Dart")

IF (ENABLE_CMAKE_TESTING)
	ENABLE_TESTING()
ENDIF (ENABLE_CMAKE_TESTING)

######################################################################################
