# --------------------------------------------------------------------------
#
#	PyEO - cmake version
#
# --------------------------------------------------------------------------
#
#	Hochschule fuer Gestaltung und Kunst Zuerich
#	Studienberreich Interaction Design
#	http://interaction.hgkz.ch
#
# --------------------------------------------------------------------------
#
#	prog:	max rheiner;xohm@users.sourceforge.net
#	date:	7/27/2007 (m/d/y)
#
# ----------------------------------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(PyEO)

FIND_PACKAGE(Boost 1.42 COMPONENTS python)
FIND_PACKAGE(PythonLibs)

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

IF(APPLE)
  # osx internal
  FIND_LIBRARY(APPLE_CARBON Carbon)
ENDIF(APPLE)

# includes
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
#INCLUDE_DIRECTORIES(../)

# source
FILE(GLOB SOURCES *.cpp)

# EO dependencies
SET(EO_SOURCES
  ${EO_SOURCE_DIR}/src/eoFunctorStore.cpp
  ${EO_SOURCE_DIR}/src/utils/eoLogger.cpp
  ${EO_SOURCE_DIR}/src/utils/eoParallel.cpp
  )

# shared library
ADD_LIBRARY(PyEO MODULE ${SOURCES} ${EO_SOURCES})
INSTALL(TARGETS PyEO LIBRARY DESTINATION local/${LIB} COMPONENT libraries)

# python 2.5 must have pyd
IF(WIN32 AND NOT CYGWIN)
  SET_TARGET_PROPERTIES(PyEO PROPERTIES SUFFIX ".pyd")
ENDIF(WIN32 AND NOT CYGWIN)

# add the libs
IF(APPLE)
  TARGET_LINK_LIBRARIES(PyEO ${APPLE_CARBON} ${PYTHON_LIBRARY} ${Boost_LIBRARIES})
ELSE(APPLE)
  TARGET_LINK_LIBRARIES(PyEO ${PYTHON_LIBRARY} ${Boost_LIBRARIES})
ENDIF(APPLE)
