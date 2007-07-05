
######################################################################################
### 0) Set the compiler and define targets to easily run the lessons
######################################################################################

SET (CMAKE_CXX_COMPILER mpicxx)

ADD_CUSTOM_TARGET(copy_lesson2_param ALL)
ADD_CUSTOM_COMMAND(
    TARGET copy_lesson2_param
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson2/lesson.param
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Lesson2)

ADD_CUSTOM_TARGET(copy_lesson2_xmlschema ALL)
ADD_CUSTOM_COMMAND(
    TARGET copy_lesson2_xmlschema
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson2/schema.xml
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Lesson2)
######################################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR})
INCLUDE_DIRECTORIES(${MO_SRC_DIR})
INCLUDE_DIRECTORIES(${PEO_SRC_DIR})
INCLUDE_DIRECTORIES(${TSP_SRC_DIR})

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries (mandatory: before 3) )
######################################################################################

LINK_DIRECTORIES(	${EO_SRC_DIR} 
					${EO_SRC_DIR}/utils 
					${ParadisEO-PEO_BINARY_DIR}/lib 
					${TSP_BINARY_DIR}/lib)
					
######################################################################################


######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

# no matter what is the OS, hopefully
ADD_EXECUTABLE(tspExample main.cpp) 

ADD_DEPENDENCIES(tspExample tsp)
ADD_DEPENDENCIES(tspExample peo)
ADD_DEPENDENCIES(tspExample rmc_mpi)

######################################################################################


######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(LESSON2_VERSION "1.0.beta")
SET_TARGET_PROPERTIES(tspExample PROPERTIES VERSION "${LESSON2_VERSION}")
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





