
######################################################################################
### 0) Set the compiler and define targets to easily run the lessons
######################################################################################

SET (CMAKE_CXX_COMPILER mpicxx)

ADD_CUSTOM_TARGET(install DEPENDS ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Meta-models/param ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Meta-models/schema.xml)

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Meta-models/param
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Meta-models)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Meta-models/schema.xml
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Meta-models)

######################################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src ${MOEO_SRC_DIR}/src ${MO_SRC_DIR}/src ${ParadisEO-PEO_SOURCE_DIR}/src ${TSP_SRC_DIR})

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${ParadisEO-PEO_BINARY_DIR}/lib ${TSP_BINARY_DIR}/lib)

######################################################################################


######################################################################################
### 3) Define the targets
######################################################################################

ADD_EXECUTABLE(multistart multistart.cpp)
ADD_DEPENDENCIES(multistart peo rmc_mpi)

ADD_EXECUTABLE(coop cooperation.cpp)
ADD_DEPENDENCIES(coop peo rmc_mpi)
######################################################################################


######################################################################################
### 4) Optionnal: define properties
######################################################################################

SET(Meta-models_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(multistart PROPERTIES VERSION "${Meta-models_VERSION}")
SET_TARGET_PROPERTIES(coop PROPERTIES VERSION "${Meta-models_VERSION}")
######################################################################################


######################################################################################
### 5) Link the librairies
######################################################################################

TARGET_LINK_LIBRARIES(multistart ${XML2_LIBS} tsp peo rmc_mpi eo eoutils peo)
TARGET_LINK_LIBRARIES(coop ${XML2_LIBS} tsp peo rmc_mpi eo eoutils peo)

######################################################################################

