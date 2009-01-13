
######################################################################################
### 0) Set the compiler and define targets to easily run the lessons
######################################################################################

SET (CMAKE_CXX_COMPILER mpicxx)

#ADD_CUSTOM_TARGET(install DEPENDS ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson6/param ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson6/schema.xml)

#ADD_CUSTOM_COMMAND(
#    TARGET install
#    POST_BUILD
#    COMMAND ${CMAKE_COMMAND}
#    ARGS -E copy_if_different
#      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson6/param
#      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Lesson6)
#ADD_CUSTOM_COMMAND(
#    TARGET install
#    POST_BUILD
#    COMMAND ${CMAKE_COMMAND}
#    ARGS -E copy_if_different
#      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson6/schema.xml
#      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Lesson6)

EXECUTE_PROCESS(
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
      	${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson6/param
      	${ParadisEO-PEO_BINARY_DIR}/tutorial/Lesson6
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
     	${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson6/schema.xml
      	${ParadisEO-PEO_BINARY_DIR}/tutorial/Lesson6
)

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
### 3) Define your target(s): just an executable here
######################################################################################

ADD_EXECUTABLE(ea6 main.cpp)
ADD_DEPENDENCIES(ea6 peo rmc_mpi)

######################################################################################


######################################################################################
### 4) Optionnal: define properties
######################################################################################

SET(Lesson6_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(ea6 PROPERTIES VERSION "${Lesson6_VERSION}")
######################################################################################


######################################################################################
### 5) Link the librairies
######################################################################################

TARGET_LINK_LIBRARIES(ea6 ${XML2_LIBS} tsp peo rmc_mpi eo eoutils peo)

######################################################################################

