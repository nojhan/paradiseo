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
	SOURCE_GROUP(src FILES FlowShopEA.cpp)
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
	ADD_EXECUTABLE(FlowShopEA 
		FlowShopEA.cpp
		${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson2/FlowShopEA.param
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
	ADD_TARGET_MOEO(lesson2)
	IF(ENABLE_CMAKE_EXAMPLE)
		ADD_EXECUTABLE(FlowShopEA FlowShopEA.cpp)
	ELSE(ENABLE_CMAKE_EXAMPLE)
		ADD_EXECUTABLE(FlowShopEA EXCLUDE_FROM_ALL FlowShopEA.cpp)
	ENDIF(ENABLE_CMAKE_EXAMPLE)
ENDIF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")

ADD_DEPENDENCIES(FlowShopEA flowshop moeo)

######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(FLOWSHOPEA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(FlowShopEA PROPERTIES VERSION "${FLOWSHOPEA_VERSION}")

######################################################################################
### 5) Link the librairies for your target(s)
######################################################################################

TARGET_LINK_LIBRARIES(FlowShopEA moeo flowshop eo eoutils)

######################################################################################
