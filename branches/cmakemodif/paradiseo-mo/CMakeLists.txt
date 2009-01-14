CMAKE_MINIMUM_REQUIRED(VERSION 2.6 FATAL_ERROR)

##########################################################################################################################################
### 0) If you want to set your own variables in mo-conf.cmake and avoid the cmd line
##########################################################################################################################################

INCLUDE(mo-conf.cmake OPTIONAL)

##########################################################################################################################################


##########################################################################################################################################
### 1) Project properties
##########################################################################################################################################

# set the project name
PROJECT(ParadisEO-MO)

SET(PACKAGE_BUGREPORT "paradiseo-help@lists.gforge.inria.fr" CACHE STRING "Package bug report" FORCE)
SET(PACKAGE_NAME "ParadisEO-MO - Moving Objects" CACHE STRING "Package name" FORCE)
SET(PACKAGE_STRING "ParadisEO-MO 1.2" CACHE STRING "MO Package string full name" FORCE)
SET(PACKAGE_VERSION "1.2" CACHE STRING "Package version" FORCE)
SET(GLOBAL_VERSION "1.2" CACHE STRING "Global version" FORCE)
SET(VERSION "1.2" CACHE STRING "Version" FORCE)
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

