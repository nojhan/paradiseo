

######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################



######################################################################################
### 2) Specify where CMake can find the libraries (mandatory: before 3) )
######################################################################################

IF(NOT WIN32 OR CYGWIN)  
  LINK_DIRECTORIES(${EO_BINARY_DIR}/lib)
ENDIF(NOT WIN32 OR CYGWIN)

# especially for Visual Studio
IF(WIN32 AND NOT CYGWIN)  
  	LINK_DIRECTORIES(${EO_BINARY_DIR}\\lib\\${CMAKE_BUILD_TYPE})
ENDIF(WIN32 AND NOT CYGWIN) 

######################################################################################



######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

SET (GPROP_SOURCES gprop.cpp)

# especially for Visual Studio
IF(NOT WIN32 OR CYGWIN)  
  	ADD_EXECUTABLE(gprop ${GPROP_SOURCES})                             
    ADD_DEPENDENCIES(gprop eo eoutils)
    
    TARGET_LINK_LIBRARIES(gprop eo eoutils) 
    
    SET(GPROP_VERSION ${GLOBAL_VERSION})
  SET_TARGET_PROPERTIES(gprop PROPERTIES VERSION "${GPROP_VERSION}")
ENDIF(NOT WIN32 OR CYGWIN)

######################################################################################

