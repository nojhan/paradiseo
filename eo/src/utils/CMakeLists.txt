######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Define the ga target
######################################################################################

SET(EOUTILS_LIB_OUTPUT_PATH ${EO_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${EOUTILS_LIB_OUTPUT_PATH})

SET(EOUTILS_SOURCES
  eoData.cpp
  eoFileMonitor.cpp
  eoGnuplot.cpp
  eoGnuplot1DMonitor.cpp
  eoGnuplot1DSnapshot.cpp
  eoIntBounds.cpp
  eoParser.cpp
  eoRealBounds.cpp
  eoRNG.cpp
  eoState.cpp
  eoOStreamMonitor.cpp
  eoUpdater.cpp
  make_help.cpp
  pipecom.cpp
  eoLogger.cpp
  eoParallel.cpp
  eoSignal.cpp
  )

ADD_LIBRARY(eoutils STATIC ${EOUTILS_SOURCES})
INSTALL(TARGETS eoutils ARCHIVE DESTINATION local/${LIB} COMPONENT libraries)


######################################################################################
### 3) Optionnal
######################################################################################

SET(EOUTILS_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(eoutils PROPERTIES VERSION "${EOUTILS_VERSION}")

######################################################################################
