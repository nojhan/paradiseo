
######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################


######################################################################################
### 2) Define the ga target
######################################################################################

SET(EOOBSOLETE_LIB_OUTPUT_PATH ${EO_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${EOOBSOLETE_LIB_OUTPUT_PATH})

SET (EOOBSOLETE_SOURCES eoParserUtils.cpp)


ADD_LIBRARY(eoobsolete STATIC ${EOOBSOLETE_SOURCES}) 
######################################################################################



######################################################################################
### 3) Optionnal
######################################################################################

SET(EOOBSOLETE_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(eoobsolete PROPERTIES VERSION "${EOOBSOLETE_VERSION}")

######################################################################################

