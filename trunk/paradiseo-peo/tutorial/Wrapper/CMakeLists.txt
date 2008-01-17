
######################################################################################
### 0) Set the compiler and define targets to easily run the lessons
######################################################################################

SET (CMAKE_CXX_COMPILER mpicxx)

ADD_CUSTOM_TARGET(install DEPENDS ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Wrapper/param ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Wrapper/schema.xml)

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Wrapper/param
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Wrapper)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Wrapper/schema.xml
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Wrapper)

######################################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src ${MOEO_SRC_DIR}/src ${MO_SRC_DIR}/src ${ParadisEO-PEO_SOURCE_DIR}/src ${TSP_SRC_DIR})

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries (mandatory: before 3) )
######################################################################################

IF(NOT WIN32 OR CYGWIN)  
  LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${MOEO_BIN_DIR}/lib ${ParadisEO-PEO_BINARY_DIR}/lib ${TSP_BINARY_DIR}/lib)
ENDIF(NOT WIN32 OR CYGWIN)

# especially for Visual Studio
IF(WIN32 AND NOT CYGWIN)  
  	LINK_DIRECTORIES(${EO_BIN_DIR}\\lib\\${CMAKE_BUILD_TYPE} ${ParadisEO-PEO_BINARY_DIR}\\lib\\${CMAKE_BUILD_TYPE} ${TSP_BINARY_DIR}\\lib\\${CMAKE_BUILD_TYPE})
ENDIF(WIN32 AND NOT CYGWIN) 

######################################################################################


######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

#ADD_EXECUTABLE(example1 main1.cpp)
#ADD_DEPENDENCIES(example1 peo rmc_mpi)
#ADD_EXECUTABLE(example2 main2.cpp)
#ADD_DEPENDENCIES(example2 peo rmc_mpi)
#ADD_EXECUTABLE(example3 main3.cpp)
#ADD_DEPENDENCIES(example3 peo rmc_mpi)

ADD_EXECUTABLE(example4 main4.cpp)
ADD_DEPENDENCIES(example4 peo rmc_mpi)
#ADD_EXECUTABLE(example5 main5.cpp)
#ADD_DEPENDENCIES(example5 tsp peo rmc_mpi)
#ADD_EXECUTABLE(example6 main6.cpp)
#ADD_DEPENDENCIES(example6 tsp peo rmc_mpi)
#ADD_EXECUTABLE(example7 main7.cpp)
#ADD_DEPENDENCIES(example7 peo rmc_mpi)

######################################################################################


######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################


######################################################################################


######################################################################################
### 5) Link the librairies
######################################################################################

#TARGET_LINK_LIBRARIES(example1 ${XML2_LIBS} peo rmc_mpi eo eoutils)
#TARGET_LINK_LIBRARIES(example2 ${XML2_LIBS} peo rmc_mpi eo eoutils)
#TARGET_LINK_LIBRARIES(example3 ${XML2_LIBS} peo rmc_mpi eo eoutils)
TARGET_LINK_LIBRARIES(example4 ${XML2_LIBS} peo rmc_mpi eo eoutils)
#TARGET_LINK_LIBRARIES(example5 ${XML2_LIBS} tsp peo rmc_mpi eo eoutils)
#TARGET_LINK_LIBRARIES(example6 ${XML2_LIBS} tsp peo rmc_mpi eo eoutils)
#TARGET_LINK_LIBRARIES(example7 ${XML2_LIBS} tsp peo rmc_mpi eo eoutils)

######################################################################################
