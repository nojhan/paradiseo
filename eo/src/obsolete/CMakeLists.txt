
######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################


######################################################################################
### 2) Define the ga target
######################################################################################

SET(EOOBSOLETE_LIB_OUTPUT_PATH ${EO_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${EOOBSOLETE_LIB_OUTPUT_PATH})

SET (EOOBSOLETE_SOURCES eoParserUtils.cpp)


ADD_LIBRARY(eoobsolete STATIC ${EOOBSOLETE_SOURCES}) 
######################################################################################



######################################################################################
### 3) Optionnal
######################################################################################

SET(EOOBSOLETE_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(eoobsolete PROPERTIES VERSION "${EOOBSOLETE_VERSION}")

######################################################################################



######################################################################################
### 4) Windows advanced config - especially for Microsoft Visual Studio 8
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

