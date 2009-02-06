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

SET(FLOWSHOP_LIB_OUTPUT_PATH ${INSTALL_DIR}/flowshop/lib)
SET(LIBRARY_OUTPUT_PATH ${FLOWSHOP_LIB_OUTPUT_PATH})

SET (FLOWSHOP_SOURCES   FlowShopBenchmarkParser.cpp 
			FlowShopEval.cpp 
			FlowShopObjectiveVectorTraits.cpp 
			FlowShopOpCrossoverQuad.cpp 
			FlowShop.cpp
			exchange_move.cpp
			exchange_move_init.cpp
			exchange_move_next.cpp
			)

ADD_LIBRARY(flowshop STATIC ${FLOWSHOP_SOURCES})

######################################################################################

