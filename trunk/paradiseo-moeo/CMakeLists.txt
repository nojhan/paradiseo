
##########################################################################################################################################
### 0) If you want to set your own variables in moeo-conf.cmake and avoid the cmd line
##########################################################################################################################################

INCLUDE(moeo-conf.cmake OPTIONAL)

##########################################################################################################################################


##########################################################################################################################################
### 1) Project properties
##########################################################################################################################################

# set the project name
PROJECT(ParadisEO-MOEO)

SET(PACKAGE_BUGREPORT "paradiseo-help@lists.gforge.inria.fr" CACHE STRING "Package bug report" FORCE)
SET(PACKAGE_NAME "ParadisEO-MOEO - Multi Objective Evolving Objects" CACHE STRING "Package name" FORCE)
SET(PACKAGE_STRING "ParadisEO-MOEO 1.1" CACHE STRING "MOEO Package string full name" FORCE)
SET(PACKAGE_VERSION "1.1" CACHE STRING "Package version" FORCE)
SET(GLOBAL_VERSION "1.1" CACHE STRING "Global version" FORCE)
SET(VERSION "1.1" CACHE STRING "Version" FORCE)
##########################################################################################################################################


##########################################################################################################################################
### 2) Include the common CMake configuration
##########################################################################################################################################

# The "config" variable must be provided on the command line
IF(NOT DEFINED config OR NOT config)
  MESSAGE(FATAL_ERROR  "The \"config\" variable must be set on the command line to 
  						give the path of the install configuration file. ")
ENDIF(NOT DEFINED config OR NOT config)

# Need the config file whose full path is given thanks to the "config" variable
INCLUDE(${config})
##########################################################################################################################################

