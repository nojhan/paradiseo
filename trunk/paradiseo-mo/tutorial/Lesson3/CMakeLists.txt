

######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src/utils)
INCLUDE_DIRECTORIES(${ParadisEO-MO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${TSP_SRC_DIR})
######################################################################################



######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

IF(NOT WIN32 OR CYGWIN)  
  LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${TSP_BINARY_DIR}/lib)
ENDIF(NOT WIN32 OR CYGWIN)

# especially for Visual Studio
IF(WIN32 AND NOT CYGWIN)  
  	LINK_DIRECTORIES(${EO_BIN_DIR}\\lib\\${CMAKE_BUILD_TYPE})
ENDIF(WIN32 AND NOT CYGWIN) 

######################################################################################



######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

ADD_EXECUTABLE(simulated_annealing simulated_annealing.cpp) 

ADD_DEPENDENCIES(simulated_annealing tsp)
######################################################################################



######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(SIMULATEDANNEALING_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(simulated_annealing PROPERTIES VERSION "${SIMULATEDANNEALING_VERSION}")
######################################################################################



######################################################################################
### 5) Link the librairies for your target(s)
######################################################################################

TARGET_LINK_LIBRARIES(simulated_annealing tsp eo eoutils) 

######################################################################################

