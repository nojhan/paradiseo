
######################################################################################
### 0) If you want to set your variables in moeo-conf.cmake and avoid the cmd line
######################################################################################

INCLUDE(moeo-conf.cmake OPTIONAL)

######################################################################################


######################################################################################
### 1) Main project config 
######################################################################################

IF(NOT DEFINED ${PROJECT_NAME} OR ${PROJECT_NAME} STREQUAL "Project")
    SET(PROJECT_NAME "ParadisEO-MOEO")
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
### 3) SET main paths (can be defined in moeo-conf.cmake)
######################################################################################

IF(NOT DEFINED MOEO_DIR)
  SET(MOEO_DIR ${CMAKE_CURRENT_SOURCE_DIR} CACHE PATH "ParadisEO-MOEO main directory") 
ENDIF(NOT DEFINED MOEO_DIR)

# Need main EO directory path
IF(NOT DEFINED EO_DIR OR NOT EO_DIR)
  SET(EO_DIR ${EOdir} CACHE PATH "ParadisEO-EO main directory" FORCE) # given on the command line
ELSE(NOT DEFINED EO_DIR OR NOT EO_DIR)
  MESSAGE(STATUS "EOdir already defined in as ${EO_DIR}")
ENDIF(NOT DEFINED EO_DIR OR NOT EO_DIR)

# Set the main paths
IF(NOT DEFINED EO_SRC_DIR)
  SET(EO_SRC_DIR "${EO_DIR}/src")
ENDIF(NOT DEFINED EO_SRC_DIR)

IF(NOT DEFINED MOEO_SRC_DIR)
  SET(MOEO_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
ENDIF(NOT DEFINED MOEO_SRC_DIR)

IF(NOT DEFINED MOEO_DOC_DIR)
  SET(MOEO_DOC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/docs)
ENDIF(NOT DEFINED MOEO_DOC_DIR)
######################################################################################


######################################################################################
### 4) Paths checking 
######################################################################################
IF(EXISTS ${EO_DIR})
	MESSAGE (STATUS "Using ParadisEO-EO path: ${EO_DIR}")
ELSE(EXISTS ${EO_DIR})
	MESSAGE (FATAL_ERROR "Could not find the ParadisEO-EO path: ${EO_DIR}")
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

SUBDIRS(docs tutorials)

######################################################################################
