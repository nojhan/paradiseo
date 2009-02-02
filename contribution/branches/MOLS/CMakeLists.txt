######################################################################################
### 1) Project properties
######################################################################################

# set the project name
PROJECT(MOLS)

# check cmake version compatibility
CMAKE_MINIMUM_REQUIRED(VERSION 2.6)

# regular expression checking
INCLUDE_REGULAR_EXPRESSION("^.*$" "^$")

# set a language
ENABLE_LANGUAGE(CXX)
ENABLE_LANGUAGE(C)

#####################################################################################


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

SUBDIRS(src test doc) 

######################################################################################

