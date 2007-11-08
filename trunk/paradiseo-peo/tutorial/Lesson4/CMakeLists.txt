
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

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src ${MO_SRC_DIR}/src ${ParadisEO-PEO_SOURCE_DIR}/src)

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

IF(NOT WIN32 OR CYGWIN)  
  LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${ParadisEO-PEO_BINARY_DIR}/lib)
ENDIF(NOT WIN32 OR CYGWIN)

# especially for Visual Studio
IF(WIN32 AND NOT CYGWIN)  
  	LINK_DIRECTORIES(${EO_BIN_DIR}\\lib\\${CMAKE_BUILD_TYPE} ${ParadisEO-PEO_BINARY_DIR}\\lib\\${CMAKE_BUILD_TYPE})
ENDIF(WIN32 AND NOT CYGWIN) 
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

SET(LESSON4_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(pso PROPERTIES VERSION "${LESSON4_VERSION}")
SET_TARGET_PROPERTIES(ea PROPERTIES VERSION "${LESSON4_VERSION}")
######################################################################################


######################################################################################
### 5) Link the librairies
######################################################################################

TARGET_LINK_LIBRARIES(pso ${XML2_LIBS} peo rmc_mpi eo eoutils) 
TARGET_LINK_LIBRARIES(ea ${XML2_LIBS} peo rmc_mpi eo eoutils)

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



