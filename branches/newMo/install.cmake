#########################################################################################################
# 1) ParadisEO install: SIMPLE Configuration
#########################################################################################################

#  Here, just specify PARADISEO_DIR : the directory where ParadisEO has been installed
SET(PARADISEO_DIR "/home/humeau/paradiseo-1.2.1" CACHE PATH "ParadisEO directory" FORCE)

#########################################################################################################
 
 
 
#########################################################################################################
# 2) ParadisEO install: ADVANCED Configuration
#########################################################################################################

SET(PARADISEO_EO_SRC_DIR "${PARADISEO_DIR}/paradiseo-eo" CACHE PATH "ParadisEO-EO source directory" FORCE)
SET(PARADISEO_EO_BIN_DIR "${PARADISEO_DIR}/paradiseo-eo/build" CACHE PATH "ParadisEO-EO binary directory" FORCE)


