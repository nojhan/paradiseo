

######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
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
ADD_EXECUTABLE(OneMaxEA OneMaxEA.cpp) 
ADD_EXECUTABLE(OneMaxLibEA OneMaxLibEA.cpp make_OneMax.cpp) 

ADD_DEPENDENCIES(OneMaxEA es ga eo eoutils)
ADD_DEPENDENCIES(OneMaxLibEA es ga eo eoutils)
######################################################################################



######################################################################################
### 4) Optionnal
######################################################################################

SET(ONEMAXEA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(OneMaxEA PROPERTIES VERSION "${ONEMAXEA_VERSION}")

SET(ONEMAXLIBEA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(OneMaxLibEA PROPERTIES VERSION "${ONEMAXLIBEA_VERSION}")
######################################################################################



######################################################################################
### 5) Link the librairies for the targets
######################################################################################

TARGET_LINK_LIBRARIES(OneMaxEA es ga eo eoutils)
TARGET_LINK_LIBRARIES(OneMaxLibEA es ga eo eoutils)

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

