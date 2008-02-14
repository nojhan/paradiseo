
######################################################################################
### 1) If you want to set your variables in peo-conf.cmake and avoid the cmd line
######################################################################################

INCLUDE(peo-conf.cmake OPTIONAL)

######################################################################################


######################################################################################
### 2) Project properties
######################################################################################

# set the project name
PROJECT(ParadisEO-PEO)

SET(PACKAGE_BUGREPORT "paradiseo-help@lists.gforge.inria.fr" CACHE STRING "Package bug report" FORCE)
SET(PACKAGE_NAME "ParadisEO-PEO - Parallel and Distributed Evolving Objects" CACHE STRING "Package name" FORCE)
SET(PACKAGE_STRING "ParadisEO-PEO 1.0" CACHE STRING "Package string full name" FORCE)
SET(PACKAGE_VERSION "1.0" CACHE STRING "Package version" FORCE)
SET(GLOBAL_VERSION "1.0" CACHE STRING "Global version" FORCE)
SET(VERSION "1.0" CACHE STRING "Version" FORCE)
#####################################################################################


#####################################################################################
### 3) Include required modules
#####################################################################################

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

#IF(WIN32)
#  SET (ABSOLUTE_PATH_REGEX "^[A-Z]:|^[a-z]:")
#ELSE(WIN32)
#  SET (ABSOLUTE_PATH_REGEX "^/")
#ENDIF(WIN32)

#SET(REQUIRED_PATHS "EO_SRC_DIR" "EO_BIN_DIR" "MO_SRC_DIR" "MO_BIN_DIR" "MOEO_SRC_DIR" "MOEO_BIN_DIR")
#FOREACH (path ${REQUIRED_PATHS})
#	  	IF(EXISTS ${${path}})
#			MESSAGE (STATUS "Using ${path}=${${path}}")
#		ELSE(EXISTS ${${path}})
#			MESSAGE (FATAL_ERROR "\n Cannot find \"${${path}}\". Please, fill \"${config}\" with a correct value for ${path} variable")
#		ENDIF(EXISTS ${${path}})
#		
#		IF(NOT ${${path}} MATCHES "${ABSOLUTE_PATH_REGEX}")
#			MESSAGE (FATAL_ERROR "${${path}} MUST BE an absolute path")
#		ENDIF(NOT ${${path}} MATCHES "${ABSOLUTE_PATH_REGEX}")
#ENDFOREACH (path ${REQUIRED_PATHS})
######################################################################################


######################################################################################
### 0) Preliminary checking: no WIN and libxml2
######################################################################################

IF(WIN32)
	MESSAGE (FATAL_ERROR "ParadisEO-PEO can not be generated on Windows systems")
ENDIF(WIN32)

# check libxml2
FIND_PROGRAM(XML2_CONFIG_PROGRAM NAMES xml2-config  PATHS /usr/local/bin /usr/bin  DOC "The libxml2 (www.xmlsoft.org) configuration tool")

IF(NOT XML2_CONFIG_PROGRAM)
	#MESSAGE(FATAL_ERROR "Libxml2 cannot be found on your system. Libxml2 is required for ParadisEO-PEO install")
ENDIF(NOT XML2_CONFIG_PROGRAM)

# set libxml2 cflags in XML2_CFLAGS
EXECUTE_PROCESS(COMMAND ${XML2_CONFIG_PROGRAM} --cflags  OUTPUT_VARIABLE XML2_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)	  

# set libxml2 libs in XML2_LIBS
EXECUTE_PROCESS(COMMAND ${XML2_CONFIG_PROGRAM} --libs  OUTPUT_VARIABLE XML2_LIBS)	  
######################################################################################


