
##########################################################################################################################################
### ParadisEO Install Configuration 
### The path to the modules EO and MO must be specified above.
##########################################################################################################################################

SET(EO_SRC_DIR "${CMAKE_SOURCE_DIR}/../paradiseo-eo" CACHE PATH "ParadisEO-EO source directory" FORCE)
SET(EO_BIN_DIR "${CMAKE_BINARY_DIR}/../../paradiseo-eo/build" CACHE PATH "ParadisEO-EO binary directory" FORCE)
    
SET(MO_SRC_DIR "${CMAKE_SOURCE_DIR}/../paradiseo-mo" CACHE PATH "ParadisMO-MO source directory" FORCE)
SET(MO_BIN_DIR "${CMAKE_BINARY_DIR}/../../paradiseo-mo/build" CACHE PATH "ParadisMO-MO binary directory" FORCE)

SET(MOEO_SRC_DIR "${CMAKE_SOURCE_DIR}/../paradiseo-moeo" CACHE PATH "ParadisMOEO-MOEO source directory" FORCE)
SET(MOEO_BIN_DIR "${CMAKE_BINARY_DIR}/../../paradiseo-moeo/build" CACHE PATH "ParadisMOEO-MOEO binary directory" FORCE)
    
###########################################################################################################################################
    
    
