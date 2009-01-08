######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src/utils)
INCLUDE_DIRECTORIES(${MO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${TSP_SRC_DIR})
######################################################################################

######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${TSP_BIN_DIR}/lib)

######################################################################################



######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

IF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008")
	SOURCE_GROUP(src FILES hill_climbing.cpp)
	
	SOURCE_GROUP(benchs FILES
		${TSP_BIN_DIR}/benchs/berlin52.tsp
		${TSP_BIN_DIR}/benchs/eil101.tsp
		${TSP_BIN_DIR}/benchs/pr2392.tsp
		${TSP_BIN_DIR}/benchs/rl5915.tsp
		${TSP_BIN_DIR}/benchs/usa13509.tsp
	)
		
	ADD_EXECUTABLE(hill_climbing
		hill_climbing.cpp
		${MO_BIN_DIR}/tutorial/Lesson2/param
		${TSP_BIN_DIR}/benchs/berlin52.tsp
		${TSP_BIN_DIR}/benchs/eil101.tsp
		${TSP_BIN_DIR}/benchs/pr2392.tsp
		${TSP_BIN_DIR}/benchs/rl5915.tsp
		${TSP_BIN_DIR}/benchs/usa13509.tsp
	)
ELSE(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008")	
	ADD_COMMANDS_MO()
	ADD_TARGET_MO(lesson2)
	IF(ENABLE_CMAKE_EXAMPLE)
		ADD_EXECUTABLE(hill_climbing hill_climbing.cpp)
	ELSE(ENABLE_CMAKE_EXAMPLE)
		ADD_EXECUTABLE(hill_climbing EXCLUDE_FROM_ALL hill_climbing.cpp)
	ENDIF(ENABLE_CMAKE_EXAMPLE)
ENDIF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008")


ADD_DEPENDENCIES(hill_climbing tsp)
######################################################################################



######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(HILLCLIMBING_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(hill_climbing PROPERTIES VERSION "${HILLCLIMBING_VERSION}")
######################################################################################



######################################################################################
### 5) Link the librairies for your target(s)
######################################################################################

TARGET_LINK_LIBRARIES(hill_climbing tsp eo eoutils) 

######################################################################################

