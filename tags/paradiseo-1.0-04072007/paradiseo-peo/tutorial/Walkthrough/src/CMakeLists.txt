
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
ADD_EXECUTABLE(exampleA exampleA.cpp) 
ADD_EXECUTABLE(exampleB exampleB.cpp) 
ADD_EXECUTABLE(exampleC exampleC.cpp) 
ADD_EXECUTABLE(exampleD exampleD.cpp) 

ADD_DEPENDENCIES(exampleA tsp)
ADD_DEPENDENCIES(exampleB tsp)
ADD_DEPENDENCIES(exampleC tsp)
ADD_DEPENDENCIES(exampleD tsp)

ADD_DEPENDENCIES(exampleA peo)
ADD_DEPENDENCIES(exampleB peo)
ADD_DEPENDENCIES(exampleC peo)
ADD_DEPENDENCIES(exampleD peo)

ADD_DEPENDENCIES(exampleA rmc_mpi)
ADD_DEPENDENCIES(exampleB rmc_mpi)
ADD_DEPENDENCIES(exampleC rmc_mpi)
ADD_DEPENDENCIES(exampleD rmc_mpi)

SET(EXECUTABLE_OUTPUT_PATH ${TUTORIAL_WALKTHROUGH_DIR}/build)
######################################################################################


######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(WALKTHROUGH_VERSION "1.0.beta")
SET_TARGET_PROPERTIES(exampleA PROPERTIES VERSION "${WALKTHROUGH_VERSION}")
SET_TARGET_PROPERTIES(exampleB PROPERTIES VERSION "${WALKTHROUGH_VERSION}")
SET_TARGET_PROPERTIES(exampleC PROPERTIES VERSION "${WALKTHROUGH_VERSION}")
SET_TARGET_PROPERTIES(exampleD PROPERTIES VERSION "${WALKTHROUGH_VERSION}")
######################################################################################


######################################################################################
### 5) Link the librairies
######################################################################################

TARGET_LINK_LIBRARIES(exampleA ${XML2_LIBS}) # define in CMakeLists.txt at PEO root dir
TARGET_LINK_LIBRARIES(exampleB ${XML2_LIBS}) # define in CMakeLists.txt at PEO root dir
TARGET_LINK_LIBRARIES(exampleC ${XML2_LIBS}) # define in CMakeLists.txt at PEO root dir
TARGET_LINK_LIBRARIES(exampleD ${XML2_LIBS}) # define in CMakeLists.txt at PEO root dir

TARGET_LINK_LIBRARIES(exampleA tsp)
TARGET_LINK_LIBRARIES(exampleB tsp)
TARGET_LINK_LIBRARIES(exampleC tsp)
TARGET_LINK_LIBRARIES(exampleD tsp)

TARGET_LINK_LIBRARIES(exampleA peo) 
TARGET_LINK_LIBRARIES(exampleB peo) 
TARGET_LINK_LIBRARIES(exampleC peo) 
TARGET_LINK_LIBRARIES(exampleD peo) 

TARGET_LINK_LIBRARIES(exampleA rmc_mpi) 
TARGET_LINK_LIBRARIES(exampleB rmc_mpi) 
TARGET_LINK_LIBRARIES(exampleC rmc_mpi) 
TARGET_LINK_LIBRARIES(exampleD rmc_mpi) 

TARGET_LINK_LIBRARIES(exampleA eo)
TARGET_LINK_LIBRARIES(exampleB eo)
TARGET_LINK_LIBRARIES(exampleC eo)
TARGET_LINK_LIBRARIES(exampleD eo)

TARGET_LINK_LIBRARIES(exampleA eoutils)
TARGET_LINK_LIBRARIES(exampleB eoutils)
TARGET_LINK_LIBRARIES(exampleC eoutils)
TARGET_LINK_LIBRARIES(exampleD eoutils)

######################################################################################
