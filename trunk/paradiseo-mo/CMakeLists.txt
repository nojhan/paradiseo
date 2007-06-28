
######################################################################################
### 0) If you want to set your variables in mo-conf.cmake and avoid the cmd line
######################################################################################

INCLUDE(mo-conf.cmake OPTIONAL)

######################################################################################


######################################################################################
### 1) Main project config 
######################################################################################

IF(NOT DEFINED ${PROJECT_NAME} OR ${PROJECT_NAME} STREQUAL "Project")
    SET(PROJECT_NAME "ParadisEO-MO")
ENDIF(NOT DEFINED ${PROJECT_NAME} OR ${PROJECT_NAME} STREQUAL "Project")

# set the project name
PROJECT(${PROJECT_NAME})

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

######################################################################################


######################################################################################
### 3) SET main paths (can be defined in mo-conf.cmake)
######################################################################################

IF(NOT DEFINED MO_DIR)
  SET(MO_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "ParadisEO-MO main directory") 
ENDIF(NOT DEFINED MO_DIR)

# Need main EO directory path
IF(NOT DEFINED EO_DIR OR NOT EO_DIR)
  SET(EO_DIR ${EOdir} CACHE PATH "ParadisEO-EO main directory" FORCE) # given on the command line
ELSE(NOT DEFINED EO_DIR OR NOT EO_DIR)
  MESSAGE(STATUS "EOdir already defined as ${EO_DIR}")
ENDIF(NOT DEFINED EO_DIR OR NOT EO_DIR)

# Set the main paths
IF(NOT DEFINED EO_SRC_DIR)
  SET(EO_SRC_DIR "${EO_DIR}/src")
ENDIF(NOT DEFINED EO_SRC_DIR)

IF(NOT DEFINED MO_SRC_DIR)
  SET(MO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
ENDIF(NOT DEFINED MO_SRC_DIR)

IF(NOT DEFINED MO_DOC_DIR)
  SET(MO_DOC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/doc)
ENDIF(NOT DEFINED MO_DOC_DIR)
######################################################################################


######################################################################################
### 4) Paths checking 
######################################################################################
IF(EXISTS ${EO_DIR})
	MESSAGE (STATUS "Using ParadisEO-EO path: ${EO_DIR}")
ELSE(EXISTS ${EO_DIR})
	MESSAGE (FATAL_ERROR "Could not find the ParadisEO-EO path: ${EO_DIR}. You should use : cmake . -DEOdir=<your path to paradiseo-eo>")
ENDIF(EXISTS ${EO_DIR})

# Is it an absolute path ?
IF(WIN32)
  SET (ABSOLUTE_PATH_REGEX "^[A-Z]:|^[a-z]:")
ELSE(WIN32)
  SET (ABSOLUTE_PATH_REGEX "^/")
ENDIF(WIN32)

IF(NOT ${EO_DIR} MATCHES "${ABSOLUTE_PATH_REGEX}")
	MESSAGE (FATAL_ERROR "${EO_DIR} MUST BE an absolute path")
ENDIF(NOT ${EO_DIR} MATCHES "${ABSOLUTE_PATH_REGEX}")

IF(WIN32)
  SET(EO_LIB_DIR "${EO_DIR}\\win\\lib\\")
ELSE(WIN32)
  SET(EO_LIB_DIR "${EO_DIR}/src")
ENDIF(WIN32)

######################################################################################


######################################################################################
### 5) Where must cmake go now ?
######################################################################################

SUBDIRS(doc tutorial)

######################################################################################
