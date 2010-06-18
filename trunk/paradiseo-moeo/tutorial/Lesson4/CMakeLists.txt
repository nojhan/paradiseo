######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${ParadisEO-MOEO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${FLOWSHOP_SRC_DIR})

######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${ParadisEO-MOEO_BINARY_DIR}/lib ${FLOWSHOP_BINARY_DIR}/lib)

######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

IF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")
	SOURCE_GROUP(src FILES FlowShopDMLS.cpp)
	SOURCE_GROUP(benchs FILES
		${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_05_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_05_02.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_10_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_10_02.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_20_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/050_05_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/050_10_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/050_20_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/100_05_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/100_10_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/100_20_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/200_10_01.txt
	)
	ADD_EXECUTABLE(FlowShopDMLS
		FlowShopDMLS.cpp
		${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson4/FlowShopDMLS.param
		${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_05_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_05_02.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_10_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_10_02.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/020_20_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/050_05_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/050_10_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/050_20_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/100_05_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/100_10_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/100_20_01.txt
    	${ParadisEO-MOEO_BINARY_DIR}/tutorial/examples/flowshop/benchs/200_10_01.txt
	)
ELSE(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")	
	ADD_COMMANDS_MOEO()
	ADD_TARGET_MOEO(lesson4)
	IF(ENABLE_CMAKE_EXAMPLE)
		ADD_EXECUTABLE(FlowShopDMLS FlowShopDMLS.cpp)
	ELSE(ENABLE_CMAKE_EXAMPLE)
		ADD_EXECUTABLE(FlowShopDMLS EXCLUDE_FROM_ALL FlowShopDMLS.cpp)
	ENDIF(ENABLE_CMAKE_EXAMPLE)
ENDIF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")

ADD_DEPENDENCIES(FlowShopDMLS flowshop moeo)

######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(FlowShopDMLS_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(FlowShopDMLS PROPERTIES VERSION "${FlowShopDMLS_VERSION}")

######################################################################################
### 5) Link the librairies for your target(s)
######################################################################################

TARGET_LINK_LIBRARIES(FlowShopDMLS moeo flowshop eo eoutils)

######################################################################################
