
######################################################################################
### 0) Copy the ESEA.param and RealEA.param files in the build directory for an easy use.
######################################################################################

ADD_CUSTOM_TARGET(param DEPENDS ${EO_SOURCE_DIR}/tutorial/Lesson4/ESEA.param)
ADD_CUSTOM_COMMAND(
    TARGET param
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
     ${EO_SOURCE_DIR}/tutorial/Lesson4/ESEA.param
     ${EO_BINARY_DIR}/tutorial/Lesson4)
    
ADD_CUSTOM_TARGET(param DEPENDS ${EO_SOURCE_DIR}/tutorial/Lesson4/RealEA.param)
ADD_CUSTOM_COMMAND(
    TARGET param
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
     ${EO_SOURCE_DIR}/tutorial/Lesson4/RealEA.param
     ${EO_BINARY_DIR}/tutorial/Lesson4) 


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src/es)
INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src/utils)
INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src/ga)
######################################################################################



######################################################################################
### 2) Specify where CMake can find the libraries
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
### 3) Define your targets
######################################################################################

# no matter what is the OS, hopefully
ADD_EXECUTABLE(BitEA BitEA.cpp) 
ADD_EXECUTABLE(RealEA RealEA.cpp) 
ADD_EXECUTABLE(ESEA ESEA.cpp) 

#ADD_DEPENDENCIES(BitEA es ga eo eoutils)
#ADD_DEPENDENCIES(RealEA es ga eo eoutils)
#ADD_DEPENDENCIES(ESEA es ga eo eoutils)
######################################################################################



######################################################################################
### 4) Optionnal
######################################################################################

SET(BITEA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(BitEA PROPERTIES VERSION "${BITEA_VERSION}")

SET(REALEA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(RealEA PROPERTIES VERSION "${REALEA_VERSION}")

SET(ESEA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(ESEA PROPERTIES VERSION "${ESEA_VERSION}")
######################################################################################



######################################################################################
### 5) Link the librairies for the targets
######################################################################################

TARGET_LINK_LIBRARIES(BitEA es ga eo eoutils)
TARGET_LINK_LIBRARIES(RealEA es ga eo eoutils)
TARGET_LINK_LIBRARIES(ESEA es ga eo eoutils)

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

