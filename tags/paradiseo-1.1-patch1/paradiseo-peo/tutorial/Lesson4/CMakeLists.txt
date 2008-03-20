
######################################################################################
### 0) Set the compiler and define targets to easily run the lessons
######################################################################################

SET (CMAKE_CXX_COMPILER mpicxx)

ADD_CUSTOM_TARGET(install DEPENDS ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson4/param ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson4/schema.xml)

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson4/param
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Lesson4)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson4/schema.xml
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Lesson4)

######################################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src ${MOEO_SRC_DIR}/src  ${MO_SRC_DIR}/src ${ParadisEO-PEO_SOURCE_DIR}/src ${TSP_SRC_DIR})

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${MOEO_BIN_DIR}/lib ${ParadisEO-PEO_BINARY_DIR}/lib ${TSP_BINARY_DIR}/lib)

######################################################################################


######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

ADD_EXECUTABLE(eals mainEALS.cpp)
ADD_DEPENDENCIES(eals tsp peo rmc_mpi)

######################################################################################


######################################################################################
### 4) Optionnal: define properties
######################################################################################

SET(LESSON4_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(eals PROPERTIES VERSION "${LESSON4_VERSION}")

######################################################################################


######################################################################################
### 5) Link the librairies
######################################################################################

TARGET_LINK_LIBRARIES(eals ${XML2_LIBS} tsp peo rmc_mpi eo eoutils peo)

######################################################################################
