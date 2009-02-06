#  Here, specify PARADISEO_DIR : the directory where ParadisEO has been installed
SET(PARADISEO_DIR "/home/jeremie/workspace/trunk" CACHE PATH "ParadisEO directory" FORCE)

#  Here, specify SOURCES_DIR : the directory where the example sources have been deposed.
SET(SOURCES_DIR "/home/jeremie/workspace/MOLS" CACHE PATH "TP sources directory, where install.cmake is" FORCE)

###########################################################################################################################################
# PLEASE DO NOT MODIFY WHAT IS BELOW
###########################################################################################################################################

### ParadisEO Install Configuration
###########################################################################################################################################
SET(PARADISEO_EO_SRC_DIR "${PARADISEO_DIR}/paradiseo-eo" CACHE PATH "ParadisEO-EO source directory" FORCE)
SET(PARADISEO_EO_BIN_DIR "${PARADISEO_DIR}/paradiseo-eo/build" CACHE PATH "ParadisEO-EO binary directory" FORCE)

SET(PARADISEO_MO_SRC_DIR "${PARADISEO_DIR}/paradiseo-mo" CACHE PATH "ParadisEO-MO source directory" FORCE)
SET(PARADISEO_MO_BIN_DIR "${PARADISEO_DIR}/paradiseo-mo/build" CACHE PATH "ParadisEO-MO binary directory" FORCE)

SET(PARADISEO_MOEO_SRC_DIR "${PARADISEO_DIR}/paradiseo-moeo" CACHE PATH "ParadisEO-MO source directory" FORCE)
SET(PARADISEO_MOEO_BIN_DIR "${PARADISEO_DIR}/paradiseo-moeo/build" CACHE PATH "ParadisEO-MOEO binary directory" FORCE)

SET(FLOWSHOP_SRC_DIR "${SOURCES_DIR}/flowshop" CACHE PATH "flowshop source directory" FORCE)

SET(INSTALL_DIR "${SOURCES_DIR}/build" CACHE PATH "Directory where the executable will be put" FORCE)

