
######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################


######################################################################################
### 2) Define your target(s): just  the moeo lib here.
######################################################################################

SET(MOEO_LIB_OUTPUT_PATH ${ParadisEO-MOEO_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${MOEO_LIB_OUTPUT_PATH})

SET (MOEO_SOURCES core/moeoObjectiveVectorTraits.cpp)

ADD_LIBRARY(moeo STATIC ${MOEO_SOURCES})

######################################################################################


######################################################################################
### 3) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(MOEO_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(moeo PROPERTIES VERSION "${MOEO_VERSION}")
######################################################################################

