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

cmake_minimum_required(VERSION 2.8)

project(PyEO)

find_package(Boost 1.42 COMPONENTS python)
find_package(PythonLibs)

include_directories(${Boost_INCLUDE_DIRS})
include_directories(${PYTHON_INCLUDE_PATH})

link_directories(${Boost_LIBRARY_DIRS})

if(APPLE)
  # osx internal
  find_library(APPLE_CARBON Carbon)
endif(APPLE)

# includes
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
#include_directories(../)

# source
file(GLOB SOURCES *.cpp)

# EO dependencies
set(EO_SOURCES
  ${EO_SRC_DIR}/src/eoFunctorStore.cpp
  ${EO_SRC_DIR}/src/utils/eoLogger.cpp
  ${EO_SRC_DIR}/src/utils/eoParallel.cpp
  )

# shared library
add_library(PyEO MODULE ${SOURCES} ${EO_SOURCES})
install(TARGETS PyEO LIBRARY DESTINATION ${LIB} COMPONENT libraries)

# python 2.5 must have pyd
if(WIN32 AND NOT CYGWIN)
  set_target_properties(PyEO PROPERTIES SUFFIX ".pyd")
endif(WIN32 AND NOT CYGWIN)

# add the libs
if(APPLE)
  target_link_libraries(PyEO ${APPLE_CARBON} ${PYTHON_LIBRARY} ${Boost_LIBRARIES})
else(APPLE)
  target_link_libraries(PyEO ${PYTHON_LIBRARY} ${Boost_LIBRARIES})
endif(APPLE)
