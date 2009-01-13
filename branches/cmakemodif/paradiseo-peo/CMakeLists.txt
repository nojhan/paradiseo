CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)

##########################################################################################################################################
### 0) If you want to set your own variables in peo-conf.cmake and avoid the cmd line
##########################################################################################################################################

INCLUDE(peo-conf.cmake OPTIONAL)

##########################################################################################################################################


######################################################################################
### 1) Preliminary checking
######################################################################################

# check libxml2
FIND_PROGRAM(XML2_CONFIG_PROGRAM NAMES xml2-config  PATHS /usr/local/bin /usr/bin  DOC "The libxml2 (www.xmlsoft.org) configuration tool")

#IF(NOT XML2_CONFIG_PROGRAM)
	#MESSAGE(FATAL_ERROR "Libxml2 cannot be found on your system. Libxml2 is required for ParadisEO-PEO install")
#ENDIF(NOT XML2_CONFIG_PROGRAM)

# set libxml2 cflags in XML2_CFLAGS
EXECUTE_PROCESS(COMMAND ${XML2_CONFIG_PROGRAM} --cflags  OUTPUT_VARIABLE XML2_CFLAGS OUTPUT_STRIP_TRAILING_WHITESPACE)	  

# set libxml2 libs in XML2_LIBSINCLUDE(FindLibXml2)
EXECUTE_PROCESS(COMMAND ${XML2_CONFIG_PROGRAM} --libs  OUTPUT_VARIABLE XML2_LIBS)	

STRING(STRIP ${XML2_LIBS} XML2_LIBS)

######################################################################################


##########################################################################################################################################
### 2) Project properties
##########################################################################################################################################

# set the project name
PROJECT(ParadisEO-PEO)

SET(PACKAGE_BUGREPORT "paradiseo-help@lists.gforge.inria.fr" CACHE STRING "Package bug report" FORCE)
SET(PACKAGE_NAME "ParadisEO-PEO - Parallel and distributed Evolving Objects" CACHE STRING "Package name" FORCE)
SET(PACKAGE_STRING "ParadisEO-PEO 1.1" CACHE STRING "PEO Package string full name" FORCE)
SET(PACKAGE_VERSION "1.1" CACHE STRING "Package version" FORCE)
SET(GLOBAL_VERSION "1.1" CACHE STRING "Global version" FORCE)
SET(VERSION "1.1" CACHE STRING "Version" FORCE)
##########################################################################################################################################


##########################################################################################################################################
### 3) Include the common CMake configuration
##########################################################################################################################################

# The "config" variable must be provided on the command line
IF(NOT DEFINED config OR NOT config)
  MESSAGE(FATAL_ERROR  "The \"config\" variable must be set on the command line to 
  						give the path of the install configuration file. ")
ENDIF(NOT DEFINED config OR NOT config)

# Need the config file whose full path is given thanks to the "config" variable
INCLUDE(${config})
##########################################################################################################################################




