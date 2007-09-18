
######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################


######################################################################################
### 2) Define the ga target
######################################################################################

SET(EOUTILS_LIB_OUTPUT_PATH ${EO_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${EOUTILS_LIB_OUTPUT_PATH})

SET (EOUTILS_SOURCES 	eoData.cpp
						eoFileMonitor.cpp
						eoGnuplot.cpp
						eoGnuplot1DMonitor.cpp
						eoGnuplot1DSnapshot.cpp
						eoIntBounds.cpp
						eoParser.cpp
						eoRealBounds.cpp
						eoRNG.cpp
						eoState.cpp
						eoStdoutMonitor.cpp
						eoUpdater.cpp
						make_help.cpp
						pipecom.cpp)


ADD_LIBRARY(eoutils STATIC ${EOUTILS_SOURCES})
######################################################################################



######################################################################################
### 3) Optionnal
######################################################################################

SET(EOUTILS_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(eoutils PROPERTIES VERSION "${EOUTILS_VERSION}")

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

