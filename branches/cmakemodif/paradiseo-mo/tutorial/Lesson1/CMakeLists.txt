######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src/utils)
INCLUDE_DIRECTORIES(${MO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${FUNCTION_SRC_DIR})

######################################################################################

######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${FUNCTION_BIN_DIR}/lib)

######################################################################################



######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

IF(ENABLE_CMAKE_COPY)
	SOURCE_GROUP(src FILES
		minimize_function.cpp
		${MO_BIN_DIR}/tutorial/Lesson1/param
		${TSP_BIN_DIR}/benchs/berlin52.tsp
		${TSP_BIN_DIR}/benchs/eil101.tsp
		${TSP_BIN_DIR}/benchs/pr2392.tsp
		${TSP_BIN_DIR}/benchs/rl5915.tsp
		${TSP_BIN_DIR}/benchs/usa13509.tsp
	)
	SOURCE_GROUP(benchs FILES
		${TSP_BIN_DIR}/benchs/berlin52.tsp
		${TSP_BIN_DIR}/benchs/eil101.tsp
		${TSP_BIN_DIR}/benchs/pr2392.tsp
		${TSP_BIN_DIR}/benchs/rl5915.tsp
		${TSP_BIN_DIR}/benchs/usa13509.tsp
	)
	ADD_EXECUTABLE(minimize_function
		minimize_function.cpp
		${MO_BIN_DIR}/tutorial/Lesson1/param
		${TSP_BIN_DIR}/benchs/berlin52.tsp
		${TSP_BIN_DIR}/benchs/eil101.tsp
		${TSP_BIN_DIR}/benchs/pr2392.tsp
		${TSP_BIN_DIR}/benchs/rl5915.tsp
		${TSP_BIN_DIR}/benchs/usa13509.tsp
	) 
ELSE(ENABLE_CMAKE_COPY)
	SOURCE_GROUP(src FILES
		minimize_function.cpp
	)
	ADD_EXECUTABLE(minimize_function
		minimize_function.cpp
	) 
ENDIF(ENABLE_CMAKE_COPY)

ADD_DEPENDENCIES(minimize_function function)

######################################################################################



######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(MINIMIZEFUNCTION_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(minimize_function PROPERTIES VERSION "${MINIMIZEFUNCTION_VERSION}")
######################################################################################



######################################################################################
### 5) Link the librairies for your target(s)
######################################################################################

TARGET_LINK_LIBRARIES(minimize_function function eo eoutils) 

######################################################################################
