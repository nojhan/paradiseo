######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Define the eo target
######################################################################################

SET(EO_LIB_OUTPUT_PATH ${EO_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${EO_LIB_OUTPUT_PATH})

SET(EO_SOURCES
  eoFunctorStore.cpp
  eoPersistent.cpp
  eoPrintable.cpp
  eoCtrlCContinue.cpp
  eoScalarFitnessAssembled.cpp
  eoSIGContinue.cpp
  )

ADD_LIBRARY(eo STATIC ${EO_SOURCES})
INSTALL(TARGETS eo ARCHIVE DESTINATION lib COMPONENT libraries)

FILE(GLOB HDRS *.h eo)
INSTALL(FILES ${HDRS} DESTINATION include/eo COMPONENT headers)

######################################################################################
### 3) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(EO_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(eo PROPERTIES VERSION "${EO_VERSION}")

######################################################################################
### 4) Where must cmake go now ?
######################################################################################

ADD_SUBDIRECTORY(do)
ADD_SUBDIRECTORY(es)
ADD_SUBDIRECTORY(ga)
ADD_SUBDIRECTORY(gp)
ADD_SUBDIRECTORY(other)
ADD_SUBDIRECTORY(utils)

IF(ENABLE_PYEO)
  ADD_SUBDIRECTORY(pyeo)
ENDIF(ENABLE_PYEO)

######################################################################################
