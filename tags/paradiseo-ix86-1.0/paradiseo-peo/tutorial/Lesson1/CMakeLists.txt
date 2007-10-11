
######################################################################################
### 0) Set the compiler,define targets to easily run the lessons and manage doc
######################################################################################

SET (CMAKE_CXX_COMPILER mpicxx)

ADD_CUSTOM_TARGET(install DEPENDS ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson1/lesson.param ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson1/schema.xml)

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson1/lesson.param
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Lesson1)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson1/schema.xml
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Lesson1)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/Lesson1/ParadisEO-PEO_Lesson1.pdf
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/Lesson1)

IF (DOXYGEN_FOUND)
        #SET(DOC_DIR ${CMAKE_BINARY_DIR}/doc CACHE PATH "PEO TSP documentation directory")
        SET(PEO_LESSON1_DOC_CONFIG_FILE "paradiseo-peo-lsn1.doxyfile" CACHE PATH "PEO tutorial lesson1 documentation configuration file")
                # define the doc target
        IF (DOXYGEN_EXECUTABLE)
                ADD_CUSTOM_TARGET(doc
                COMMAND ${DOXYGEN_EXECUTABLE} ${PEO_LESSON1_DOC_CONFIG_FILE}
                WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
            )
        ENDIF (DOXYGEN_EXECUTABLE)

        # configure doxyfile file
        CONFIGURE_FILE("${CMAKE_SOURCE_DIR}/tutorial/Lesson1/${PEO_LESSON1_DOC_CONFIG_FILE}.cmake"
                       "${CMAKE_BINARY_DIR}/tutorial/Lesson1/${PEO_LESSON1_DOC_CONFIG_FILE}")

ELSE (DOXYGEN_FOUND)
        MESSAGE(STATUS "Unable to generate the documentation, Doxygen package not found")
ENDIF (DOXYGEN_FOUND)
######################################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src ${MO_SRC_DIR}/src ${ParadisEO-PEO_SOURCE_DIR}/src ${TSP_SRC_DIR})

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
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

ADD_EXECUTABLE(tspExample main.cpp) 
ADD_DEPENDENCIES(tspExample tsp peo rmc_mpi)
######################################################################################


######################################################################################
### 4) Optionnal: define properties
######################################################################################

SET(LESSON1_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(tspExample PROPERTIES VERSION "${LESSON1_VERSION}")
######################################################################################


######################################################################################
### 5) Link the librairies
######################################################################################

TARGET_LINK_LIBRARIES(tspExample ${XML2_LIBS} tsp peo rmc_mpi eo eoutils) 

######################################################################################


######################################################################################
### 6) Windows advanced config - especially for Microsoft Visual Studio 8
######################################################################################

  IF(CMAKE_CXX_COMPILER MATCHES cl)
   IF(NOT WITH_SHARED_LIBS)
     IF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005")
       SET(CMAKE_CXX_FLAGS "/nologo /W3 /Gy")
       SET(CMAKE_CXX_FLAGS_DEBUG "/MTd /Z7 /Od")
       SET(CMAKE_CXX_FLAGS_RELEASE "/MT /O2")
       SET(CMAKE_CXX_FLAGS_MINSIZEREL "/MT /O2")
       SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MTd /Z7 /Od")
       SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")
       
     ENDIF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005")
   ENDIF(NOT WITH_SHARED_LIBS)
  ENDIF(CMAKE_CXX_COMPILER MATCHES cl)
######################################################################################



