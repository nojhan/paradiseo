######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${PARADISEO_EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${PARADISEO_MOEO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${PARADISEO_MO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################


######################################################################################
### 2) Define your target(s): just the flow-shop lib here
######################################################################################

SET(DTLZ_LIB_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${DTLZ_LIB_OUTPUT_PATH})

SET (DTLZ_SOURCES
		DTLZ.cpp
		DTLZ1Eval.cpp
		DTLZ2Eval.cpp
		DTLZ3Eval.cpp
		DTLZ4Eval.cpp
		DTLZ5Eval.cpp
		DTLZ6Eval.cpp
		DTLZ7Eval.cpp)

ADD_LIBRARY(lDTLZ STATIC ${DTLZ_SOURCES})

######################################################################################

