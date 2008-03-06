
######################################################################################
### 0) Copy the "benchs" directory in the build directory to easily run the lessons
######################################################################################

ADD_CUSTOM_TARGET(install DEPENDS ${FLOWSHOP_SRC_DIR}/benchs)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_directory 
      ${FLOWSHOP_SRC_DIR}/benchs
      ${FLOWSHOP_BINARY_DIR}/benchs)

######################################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${ParadisEO-MOEO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${FLOWSHOP_SRC_DIR})

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${EO_BIN_DIR}/lib)

######################################################################################


######################################################################################
### 3) Define your target(s): just the flowshop lib here
######################################################################################

SET(FLOWSHOP_LIB_OUTPUT_PATH ${FLOWSHOP_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${FLOWSHOP_LIB_OUTPUT_PATH})

SET (FLOWSHOP_SOURCES   FlowShopBenchmarkParser.cpp 
			FlowShopEval.cpp 
			FlowShopObjectiveVectorTraits.cpp 
			FlowShopOpCrossoverQuad.cpp 
			FlowShop.cpp)

ADD_LIBRARY(flowshop STATIC ${FLOWSHOP_SOURCES})
ADD_DEPENDENCIES(flowshop moeo)
######################################################################################



######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(FLOWSHOP_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(flowshop PROPERTIES VERSION "${FLOWSHOP_VERSION}")
######################################################################################


######################################################################################
### 5) Link the librairies for your target(s)
######################################################################################

TARGET_LINK_LIBRARIES(flowshop eo)

######################################################################################
