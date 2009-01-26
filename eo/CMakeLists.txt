######################################################################################
### 0) If you want to set your variables in eo-conf.cmake and avoid the cmd line
######################################################################################

INCLUDE(eo-conf.cmake OPTIONAL)

######################################################################################
### 1) Main project config 
######################################################################################

# set the project name and other variables
PROJECT(EO)

SET(PACKAGE_BUGREPORT "eodev-help@sourceforge.net" CACHE STRING "Package bug report" FORCE)
SET(PACKAGE_NAME "Evolving Objects" CACHE STRING "Package name" FORCE)
SET(PACKAGE_STRING "Evolving Objects 1.02-cvs1"CACHE STRING "Package string full name" FORCE)
SET(PACKAGE_TARNAME "eo" CACHE STRING "Package tar name" FORCE)
SET(PACKAGE_VERSION "1.02-cvs1" CACHE STRING "Package version" FORCE)
SET(VERSION "1.02" CACHE STRING "Global version" FORCE)
SET(GLOBAL_VERSION "${VERSION}")

# check cmake version compatibility
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# regular expression checking
INCLUDE_REGULAR_EXPRESSION("^.*$" "^$")

# set a language for the entire project.
ENABLE_LANGUAGE(CXX)
ENABLE_LANGUAGE(C)

#####################################################################################
### 2) Include required modules / configuration files
#####################################################################################

INCLUDE(CMakeBackwardCompatibilityCXX)

INCLUDE(FindDoxygen)

INCLUDE(FindGnuplot)

INCLUDE(CheckLibraryExists)

INCLUDE(ConfigureChecks.cmake)

INCLUDE(Dart OPTIONNAL)         
                           
# now create config headers
configure_file(config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/config.h)

# Set a special flag if the environment is windows (should do the same in a config.g file)
IF (WIN32)
	ADD_DEFINITIONS(-D_WINDOWS=1)
ENDIF (WIN32)
    
#####################################################################################
### 3) Manage the build type
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
     IF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008")
       SET(CMAKE_CXX_FLAGS "/nologo /W3 /Gy")
       SET(CMAKE_CXX_FLAGS_DEBUG "/MTd /Z7 /Od")
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

IF(CMAKE_BUILD_TYPE MATCHES Debug)
    ADD_DEFINITIONS(-DCMAKE_VERBOSE_MAKEFILE=ON)
ENDIF(CMAKE_BUILD_TYPE MATCHES Debug)

######################################################################################
### compilation of examples?
######################################################################################

SET(ENABLE_CMAKE_EXAMPLE TRUE CACHE BOOL "Enable copy of benchs and parameters file?")

######################################################################################
### 4) Test config
######################################################################################

IF (ENABLE_CMAKE_TESTING)  
        ENABLE_TESTING()          
ENDIF (ENABLE_CMAKE_TESTING)

######################################################################################
### 5) Where must cmake go now ?
######################################################################################

ADD_SUBDIRECTORY(app)
ADD_SUBDIRECTORY(doc)
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(tutorial)

######################################################################################
