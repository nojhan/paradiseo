
######################################################################################
### 0) Set the compiler and define targets to easily run the lessons
######################################################################################

SET (CMAKE_CXX_COMPILER mpicxx)

ADD_CUSTOM_TARGET(install DEPENDS ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson6/param ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson6/schema.xml)

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson6/param
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Lesson6)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson6/schema.xml
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Lesson6)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson6/Lesson6.pdf
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Lesson6)
######################################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src ${MO_SRC_DIR}/src ${ParadisEO-PEO_SOURCE_DIR}/src ${TSP_SRC_DIR})

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries (mandatory: before 3) )
######################################################################################

IF(NOT WIN32 OR CYGWIN)  
  LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${ParadisEO-PEO_BINARY_DIR}/lib ${TSP_BINARY_DIR}/lib)
ENDIF(NOT WIN32 OR CYGWIN)

# especially for Visual Studio
IF(WIN32 AND NOT CYGWIN)  
  	LINK_DIRECTORIES(${EO_BIN_DIR}\\lib\\${CMAKE_BUILD_TYPE} ${ParadisEO-PEO_BINARY_DIR}\\lib\\${CMAKE_BUILD_TYPE} ${TSP_BINARY_DIR}\\lib\\${CMAKE_BUILD_TYPE})
ENDIF(WIN32 AND NOT CYGWIN) 

######################################################################################


######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

ADD_EXECUTABLE(example main.cpp)
ADD_DEPENDENCIES(example tsp peo rmc_mpi)

######################################################################################


######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(LESSON6_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(example PROPERTIES VERSION "${LESSON6_VERSION}")

######################################################################################


######################################################################################
### 5) Link the librairies
######################################################################################

TARGET_LINK_LIBRARIES(example ${XML2_LIBS} tsp peo rmc_mpi eo eoutils)


######################################################################################





