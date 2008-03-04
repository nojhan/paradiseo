######################################################################################
### 0) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${ParadisEO-MO_SOURCE_DIR}/src)

######################################################################################


######################################################################################
### 1) Define your target(s): just the tsp here
######################################################################################

SET(FUNCTION_LIB_OUTPUT_PATH ${FUNCTION_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${FUNCTION_LIB_OUTPUT_PATH})

SET (FUNCTION_SOURCES affectation_eval.cpp 
    		      affectation_init.cpp
		      deviation.cpp
		      deviation_init.cpp
		      deviation_next.cpp
		      deviation_incr_eval.cpp )
					
ADD_LIBRARY(function STATIC ${FUNCTION_SOURCES})

######################################################################################


######################################################################################
### 2) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(FUNCTION_VERSION "${GLOBAL_VERSION}")
SET_TARGET_PROPERTIES(function PROPERTIES VERSION "${FUNCTION_VERSION}")
######################################################################################
