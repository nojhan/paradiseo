
######################################################################################
### 0) Preliminary checking: no WIN and libxml2
######################################################################################

IF(WIN32)
	#MESSAGE (FATAL_ERROR "ParadisEO-PEO can not be generated on Windows systems")
ENDIF(WIN32)

# check libxml2
FIND_PROGRAM(XML2_CONFIG_PROGRAM NAMES xml2-config  PATHS /usr/local/bin /usr/bin  DOC "The libxml2 (www.xmlsoft.org) configuration tool")

IF(NOT XML2_CONFIG_PROGRAM)
	MESSAGE(FATAL_ERROR "Libxml2 cannot be found on your system. Libxml2 is required for ParadisEO-PEO install")
ENDIF(NOT XML2_CONFIG_PROGRAM)

# set libxml2 cflags in XML2_CFLAGS
EXECUTE_PROCESS(COMMAND ${XML2_CONFIG_PROGRAM} --cflags  OUTPUT_VARIABLE XML2_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)	  
SET(XML2_CFLAGS_WITH_WHITESPACE " ${XML2_CFLAGS}")

# set libxml2 libs in XML2_LIBS
EXECUTE_PROCESS(COMMAND ${XML2_CONFIG_PROGRAM} --libs  OUTPUT_VARIABLE XML2_LIBS OUTPUT_STRIP_TRAILING_WHITESPACE)	  
######################################################################################


######################################################################################
### 1) If you want to set your variables in moeo-conf.cmake and avoid the cmd line
######################################################################################

INCLUDE(peo-conf.cmake OPTIONAL)

######################################################################################


######################################################################################
### 2) Main project config 
######################################################################################

# set the project name
PROJECT(ParadisEO-PEO)

# check cmake version compatibility
CMAKE_MINIMUM_REQUIRED(VERSION 2.4 FATAL_ERROR)

# regular expression checking
INCLUDE_REGULAR_EXPRESSION("^.*$" "^$")

# set a language for the entire project.
ENABLE_LANGUAGE(CXX)
ENABLE_LANGUAGE(C)

#####################################################################################


#####################################################################################
### 3) Include required modules
#####################################################################################
INCLUDE(CMakeBackwardCompatibilityCXX)

INCLUDE(FindDoxygen)

INCLUDE(FindGnuplot)

# check for some functions
INCLUDE(CheckLibraryExists)

######################################################################################


######################################################################################
### 4) SET main paths (can be defined in peo-conf.cmake)
######################################################################################

IF(NOT DEFINED PEO_DIR)
  SET(PEO_DIR ${ParadisEO-PEO_SOURCE_DIR} CACHE PATH "ParadisEO-PEO main directory") 
ENDIF(NOT DEFINED PEO_DIR)

# Need main EO directory path
IF(NOT DEFINED EO_DIR OR NOT EO_DIR)
  SET(EO_DIR ${EOdir} CACHE PATH "ParadisEO-EO main directory" FORCE) # given on the command line
ELSE(NOT DEFINED EO_DIR OR NOT EO_DIR)
  MESSAGE(STATUS "EOdir already defined as ${EO_DIR}")
ENDIF(NOT DEFINED EO_DIR OR NOT EO_DIR)

# Set the main paths for EO
IF(NOT DEFINED EO_SRC_DIR)
  SET(EO_SRC_DIR "${EO_DIR}/src")
ENDIF(NOT DEFINED EO_SRC_DIR)

# Need main MO directory path
IF(NOT DEFINED MO_DIR OR NOT MO_DIR)
  SET(MO_DIR ${MOdir} CACHE PATH "ParadisMO-MO main directory" FORCE) # given on the command line
ELSE(NOT DEFINED MO_DIR OR NOT MO_DIR)
  MESSAGE(STATUS "MOdir already defined as ${MO_DIR}")
ENDIF(NOT DEFINED MO_DIR OR NOT MO_DIR)

# Set the main paths for MO
IF(NOT DEFINED MO_SRC_DIR)
  SET(MO_SRC_DIR "${MO_DIR}/src")
ENDIF(NOT DEFINED MO_SRC_DIR)

IF(NOT DEFINED PEO_SRC_DIR)
  SET(PEO_SRC_DIR ${PEO_DIR}/src)
ENDIF(NOT DEFINED PEO_SRC_DIR)

IF(NOT DEFINED PEO_DOC_DIR)
  SET(PEO_DOC_DIR ${PEO_DIR}/doc)
ENDIF(NOT DEFINED PEO_DOC_DIR)
######################################################################################


######################################################################################
### 5) Paths checking 
######################################################################################
IF(EXISTS ${EO_DIR})
	MESSAGE (STATUS "Using ParadisEO-EO path: ${EO_DIR}")
ELSE(EXISTS ${EO_DIR})
	MESSAGE (FATAL_ERROR "Could not find the ParadisEO-EO path: ${EO_DIR}. You should use : cmake . -DEOdir=<your path to paradiseo-eo> -DMOdir=<your path to paradiseo-mo>")
ENDIF(EXISTS ${EO_DIR})

# Is it an absolute path ?
SET (ABSOLUTE_PATH_REGEX "^/")
IF(NOT ${EO_DIR} MATCHES "${ABSOLUTE_PATH_REGEX}")
	MESSAGE (FATAL_ERROR "${EO_DIR} MUST BE an absolute path")
ENDIF(NOT ${EO_DIR} MATCHES "${ABSOLUTE_PATH_REGEX}")

SET(EO_LIB_DIR "${EO_DIR}/src")


IF(EXISTS ${MO_DIR})
	MESSAGE (STATUS "Using ParadisMO-MO path: ${MO_DIR}")
ELSE(EXISTS ${MO_DIR})
	MESSAGE (FATAL_ERROR "Could not find the ParadisMO-MO path: ${MO_DIR}. You should use : cmake . -DEOdir=<your path to paradiseo-eo> -DMOdir=<your path to paradiseo-mo>")
ENDIF(EXISTS ${MO_DIR})

# Is it an absolute path ?
SET (ABSOLUTE_PATH_REGEX "^/")
IF(NOT ${MO_DIR} MATCHES "${ABSOLUTE_PATH_REGEX}")
	MESSAGE (FATAL_ERROR "${MO_DIR} MUST BE an absolute path")
ENDIF(NOT ${MO_DIR} MATCHES "${ABSOLUTE_PATH_REGEX}")
######################################################################################


######################################################################################
### 6) Where must cmake go now ?
######################################################################################

SUBDIRS(doc src ) #tutorial)

######################################################################################
