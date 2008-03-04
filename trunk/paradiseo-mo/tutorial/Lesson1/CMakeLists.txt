######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src/utils)
INCLUDE_DIRECTORIES(${ParadisEO-MO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${FUNCTION_SRC_DIR})

######################################################################################

ADD_CUSTOM_TARGET(install DEPENDS ${ParadisEO-MO_SOURCE_DIR}/tutorial/Lesson1/param )

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-MO_SOURCE_DIR}/tutorial/Lesson1/param
      ${ParadisEO-MO_BINARY_DIR}/tutorial/Lesson1)

######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

IF(NOT WIN32 OR CYGWIN)  
  LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${FUNCTION_BINARY_DIR}/lib)
ENDIF(NOT WIN32 OR CYGWIN)

# especially for Visual Studio
IF(WIN32 AND NOT CYGWIN)  
  	LINK_DIRECTORIES(${EO_BIN_DIR}\\lib\\${CMAKE_BUILD_TYPE})
ENDIF(WIN32 AND NOT CYGWIN) 

######################################################################################



######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

ADD_EXECUTABLE(minimize_function minimize_function.cpp) 

ADD_DEPENDENCIES(minimize_function function)

######################################################################################



######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(MINIMIZEFUNCTION_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(minimize_function PROPERTIES VERSION "${MINIMIZEFUNCTION_VERSION}")
######################################################################################



######################################################################################
### 5) Link the librairies for your target(s)
######################################################################################

TARGET_LINK_LIBRARIES(minimize_function function eo eoutils) 

######################################################################################
