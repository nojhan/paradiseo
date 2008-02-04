
######################################################################################
### 0) Set the compiler and define targets to easily run the lessons
######################################################################################

SET (CMAKE_CXX_COMPILER mpicxx)
SET(MAKE_SRC_DIR "${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson8/make" CACHE PATH "Make source directory" FORCE)

ADD_CUSTOM_TARGET(install DEPENDS ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson8/param ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson8/schema.xml)

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson8/param
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Lesson8)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson8/schema.xml
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Lesson8)

######################################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src ${MOEO_SRC_DIR}/src ${FLOWSHOP_SRC_DIR} ${MO_SRC_DIR}/src ${ParadisEO-PEO_SOURCE_DIR}/src ${TSP_SRC_DIR} ${MAKE_SRC_DIR})

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries (mandatory: before 3) )
######################################################################################

IF(NOT WIN32 OR CYGWIN)  
  LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${MOEO_BIN_DIR}/lib ${FLOWSHOP_BIN_DIR}/lib ${ParadisEO-PEO_BINARY_DIR}/lib ${TSP_BINARY_DIR}/lib)
ENDIF(NOT WIN32 OR CYGWIN)

 

######################################################################################


######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

ADD_EXECUTABLE(example main.cpp)
ADD_DEPENDENCIES(example flowshop moeo peo rmc_mpi)


######################################################################################


######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################


######################################################################################

######################################################################################
### 5) Link the librairies
######################################################################################

TARGET_LINK_LIBRARIES(example ${XML2_LIBS} flowshop moeo peo rmc_mpi eo eoutils)

######################################################################################
