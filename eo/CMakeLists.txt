
######################################################################################
### 0) If you want to set your variables in eo-conf.cmake and avoid the cmd line
######################################################################################

INCLUDE(eo-conf.cmake OPTIONAL)

######################################################################################
 


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
CMAKE_MINIMUM_REQUIRED(VERSION 2.4 FATAL_ERROR)

# regular expression checking
INCLUDE_REGULAR_EXPRESSION("^.*$" "^$")

# set a language for the entire project.
ENABLE_LANGUAGE(CXX)
ENABLE_LANGUAGE(C)

####################################################################################


#####################################################################################
### 2) Include required modules
#####################################################################################
INCLUDE(CMakeBackwardCompatibilityCXX)

INCLUDE(FindDoxygen)

INCLUDE(FindGnuplot)

INCLUDE(CheckLibraryExists)

INCLUDE(ConfigureChecks.cmake)

# now create config headers
configure_file(config.h.cmake ${CMAKE_CURRENT_BINARY_DIR}/config.h )

# Set a special flag if the environment is windows (should do the same in a config.g file)
IF (WIN32)
	ADD_DEFINITIONS(-D_WINDOWS=1)
ENDIF (WIN32)
######################################################################################


#####################################################################################
### 3) Manage the build type
#####################################################################################

# the user should choose the build type on windows environments,excepted under cygwin (default=none)
SET(CMAKE_DEFAULT_BUILD_TYPE "" CACHE STRING "Variable that stores the default CMake build type" FORCE)

IF(WIN32 AND NOT CYGWIN)
  IF(NOT CMAKE_BUILD_TYPE)
     SET( CMAKE_BUILD_TYPE 
          ${CMAKE_DEFAULT_BUILD_TYPE} CACHE STRING 
          "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel." 
           FORCE)
  ENDIF(NOT CMAKE_BUILD_TYPE)  
  MESSAGE(STATUS "")
  MESSAGE(STATUS "Warning: The type of build is: ${CMAKE_BUILD_TYPE}.")
  MESSAGE(STATUS "The available types are: None Debug Release RelWithDebInfo MinSizeRel.")
  MESSAGE(STATUS "You can choose it with: cmake <path-to-source> -D<build-type>")
  MESSAGE(STATUS "")
ENDIF(WIN32 AND NOT CYGWIN)

#####################################################################################


######################################################################################
### 3) Where must cmake go now ?
######################################################################################

SUBDIRS(app doc src test tutorial)

######################################################################################


######################################################################################
### 4) Test config
######################################################################################

#SET(ENABLE_CMAKE_TESTING TRUE CACHE BOOL "Enable testing ?")

IF (ENABLE_CMAKE_TESTING)
	ENABLE_TESTING()
ENDIF (ENABLE_CMAKE_TESTING)

######################################################################################

