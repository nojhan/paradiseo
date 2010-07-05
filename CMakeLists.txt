######################################################################################
### 1) Set the application properties
######################################################################################

# Checks cmake version compatibility
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

PROJECT(DO)

SET(PROJECT_VERSION_MAJOR 1)
SET(PROJECT_VERSION_MINOR 0)
SET(PROJECT_VERSION_PATCH 0)
SET(PROJECT_VERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}")

######################################################################################


######################################################################################
### 2) Include the install configuration file where are defined the main variables
######################################################################################

INCLUDE(${CMAKE_CURRENT_SOURCE_DIR}/install.cmake)

######################################################################################


#####################################################################################
### 3) Include required modules & utilities
#####################################################################################

INCLUDE(CMakeBackwardCompatibilityCXX)
#INCLUDE(FindDoxygen)
INCLUDE(CheckLibraryExists)
#INCLUDE(Dart OPTIONAL)

FIND_PACKAGE(Doxygen)
FIND_PACKAGE(MPI REQUIRED)
FIND_PACKAGE(Gnuplot REQUIRED)

# Set a special flag if the environment is windows (should do the same in a config.g file)
IF (WIN32)
  ADD_DEFINITIONS(-D_WINDOWS=1)
ENDIF (WIN32)

######################################################################################


#####################################################################################
### 4) Manage the build type
#####################################################################################

INCLUDE(${CMAKE_SOURCE_DIR}/BuildConfig.cmake)

######################################################################################


######################################################################################
### 5) Now where we go ???
######################################################################################

# warning: you have to compile libraries before bopo sources because of dependencies.
ADD_SUBDIRECTORY(lib)
ADD_SUBDIRECTORY(doc)
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(application)

######################################################################################


######################################################################################
### 6) Include packaging
######################################################################################

INCLUDE(Packaging.cmake)

######################################################################################
