######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${PARADISEO_EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${PARADISEO_MOEO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################


######################################################################################
### 2) Define your target(s): just the flow-shop lib here
######################################################################################

SET(flow-shop_LIB_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${flow-shop_LIB_OUTPUT_PATH})

SET (flow-shop_SOURCES	FlowShopBenchmarkParser.cpp 
			FlowShopEval.cpp 
			FlowShopInit.cpp 
			FlowShopObjectiveVectorTraits.cpp 
			FlowShopOpCrossoverQuad.cpp 
			FlowShopOpMutationExchange.cpp 
			FlowShopOpMutationShift.cpp 
			FlowShop.cpp)

ADD_LIBRARY(lflowshop STATIC ${flow-shop_SOURCES})

######################################################################################

