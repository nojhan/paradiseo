###############################################################################
# 1) ParadisEO install: SIMPLE Configuration
###############################################################################


# Here, just specify PARADISEO_DIR : the directory where ParadisEO has
# been installed
SET(PARADISEO_DIR "${CMAKE_BINARY_DIR}/../../../" CACHE PATH "ParadisEO directory" FORCE)

###############################################################################
 
 
 
###############################################################################
# 2) ParadisEO install: ADVANCED Configuration
###############################################################################

SET(PARADISEO_EO_SRC_DIR "${PARADISEO_DIR}/eo" CACHE PATH "ParadisEO-EO source directory" FORCE)
SET(PARADISEO_EO_BIN_DIR "${PARADISEO_DIR}/build/eo" CACHE PATH "ParadisEO-EO binary directory" FORCE)

SET(PARADISEO_MO_SRC_DIR "${PARADISEO_DIR}/mo" CACHE PATH "ParadisEO-MO source directory" FORCE)
SET(PARADISEO_MO_BIN_DIR "${PARADISEO_DIR}/build/mo" CACHE PATH "ParadisEO-MO binary directory" FORCE)

SET(PARADISEO_MOEO_SRC_DIR "${PARADISEO_DIR}/moeo" CACHE PATH "ParadisEO-MOEO source directory" FORCE)
SET(PARADISEO_MOEO_BIN_DIR "${PARADISEO_DIR}/build/moeo" CACHE PATH "ParadisEO-MOEO binary directory" FORCE)

###############################################################################

