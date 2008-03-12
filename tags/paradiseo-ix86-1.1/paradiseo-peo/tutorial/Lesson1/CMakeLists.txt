
######################################################################################
### 0) Set the compiler and define targets to easily run the lessons
######################################################################################

SET (CMAKE_CXX_COMPILER mpicxx)

ADD_CUSTOM_TARGET(install DEPENDS ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson1/param ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson1/schema.xml)

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson1/param
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Lesson1)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson1/schema.xml
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Lesson1)

######################################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src ${MOEO_SRC_DIR}/src ${MO_SRC_DIR}/src ${ParadisEO-PEO_SOURCE_DIR}/src)

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${ParadisEO-PEO_BINARY_DIR}/lib)

######################################################################################


######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

ADD_EXECUTABLE(pso mainPSO.cpp)
ADD_DEPENDENCIES(pso peo rmc_mpi) 
ADD_EXECUTABLE(ea mainEA.cpp)
ADD_DEPENDENCIES(ea peo rmc_mpi)
######################################################################################


######################################################################################
### 4) Optionnal: define properties
######################################################################################

SET(LESSON1_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(pso PROPERTIES VERSION "${LESSON1_VERSION}")
SET_TARGET_PROPERTIES(ea PROPERTIES VERSION "${LESSON1_VERSION}")
######################################################################################


######################################################################################
### 5) Link the librairies
######################################################################################

TARGET_LINK_LIBRARIES(pso ${XML2_LIBS} peo rmc_mpi eo eoutils peo) 
TARGET_LINK_LIBRARIES(ea ${XML2_LIBS} peo rmc_mpi eo eoutils peo)

######################################################################################
