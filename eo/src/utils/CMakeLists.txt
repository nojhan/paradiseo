######################################################################################
### 1) Include the sources
######################################################################################

include_directories(${EO_SRC_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Define the ga target
######################################################################################

set(EOUTILS_LIB_OUTPUT_PATH ${EO_BIN_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${EOUTILS_LIB_OUTPUT_PATH})

set(EOUTILS_SOURCES
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

add_library(eoutils STATIC ${EOUTILS_SOURCES})
install(TARGETS eoutils EXPORT paradiseo-targets ARCHIVE DESTINATION ${LIB} COMPONENT libraries)


######################################################################################
### 3) Optionnal
######################################################################################

set(EOUTILS_VERSION ${GLOBAL_VERSION})
set_target_properties(eoutils PROPERTIES VERSION "${EOUTILS_VERSION}")

######################################################################################
