#########################################################################################################
# 1) ParadisEO install: SIMPLE Configuration
#########################################################################################################

IF(NOT DEFINED paradis OR NOT paradis)
  MESSAGE(FATAL_ERROR  "The \"paradis\" variable must be set on the command line to 
  						give the path of the install configuration file. ")
ENDIF(NOT DEFINED config OR NOT config)

#  Here, just specify PARADISEO_DIR : the directory where ParadisEO has been installed
SET(PARADISEO_DIR "${paradis}" CACHE PATH "ParadisEO directory" FORCE)

#########################################################################################################
 
 
 
#########################################################################################################
# 2) ParadisEO install: ADVANCED Configuration
#########################################################################################################

SET(PARADISEO_EO_SRC_DIR "${PARADISEO_DIR}/paradiseo-eo" CACHE PATH "ParadisEO-EO source directory" FORCE)
SET(PARADISEO_EO_BIN_DIR "${PARADISEO_DIR}/paradiseo-eo/build" CACHE PATH "ParadisEO-EO binary directory" FORCE)

SET(NEWMO_SRC_DIR "${CMAKE_SOURCE_DIR}" CACHE PATH "ParadisEO-EO source directory" FORCE)
SET(NEWMO_BIN_DIR "${CMAKE_SOURCE_DIR}" CACHE PATH "ParadisEO-EO binary directory" FORCE)


