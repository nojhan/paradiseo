
######################################################################################
### 0) Set the compiler
######################################################################################

SET (CMAKE_CXX_COMPILER mpicxx)

######################################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR})
INCLUDE_DIRECTORIES(${MO_SRC_DIR})
INCLUDE_DIRECTORIES(${PEO_SRC_DIR})
INCLUDE_DIRECTORIES(${TSP_EXAMPLE_DIR}/src)

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries (mandatory: before 3) )
######################################################################################

LINK_DIRECTORIES(	${EO_SRC_DIR} 
					${EO_SRC_DIR}/utils 
					${PEO_DIR}/build 
					${TSP_EXAMPLE_DIR}/build)

######################################################################################


######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

# no matter what is the OS, hopefully
ADD_EXECUTABLE(tspExample main.cpp) 

ADD_DEPENDENCIES(tspExample tsp)
ADD_DEPENDENCIES(tspExample peo)
ADD_DEPENDENCIES(tspExample rmc_mpi)

SET(EXECUTABLE_OUTPUT_PATH ${TUTORIAL_LESSON1_DIR}/build)
######################################################################################


######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(LESSON1_VERSION "1.0.beta")
SET_TARGET_PROPERTIES(tspExample PROPERTIES VERSION "${LESSON1_VERSION}")
######################################################################################


######################################################################################
### 5) Link the librairies
######################################################################################

TARGET_LINK_LIBRARIES(tspExample ${XML2_LIBS}) # define in CMakeLists.txt at PEO root dir
TARGET_LINK_LIBRARIES(tspExample tsp)
TARGET_LINK_LIBRARIES(tspExample peo) 
TARGET_LINK_LIBRARIES(tspExample rmc_mpi) 
TARGET_LINK_LIBRARIES(tspExample eo)
TARGET_LINK_LIBRARIES(tspExample eoutils)

######################################################################################





