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

# change this to your local boost-path 
# SET(BOOST_PATH "C:/development/libs/os/boost/boost_1_33_1")

# ----------------------------------------------------------------------------

CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

PROJECT(PyEO)

FIND_PACKAGE(Boost 1.42 COMPONENTS program_options python)

INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIRS})
LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

# python
IF(APPLE)

	# python
	# osx can have several version of python, on dif places
#	SET(PYTHON /Library/Frameworks/Python.framework/Versions/)
#	SET(PYTHON_VERSION 2.4)
#	SET(PYTHON_INCLUDE_PATH "${PYTHON}${PYTHON_VERSION}/include/python${PYTHON_VERSION}" CACHE STRING "")
#	SET(PYTHON_BINARY ${PYTHON}${PYTHON_VERSION}/bin/python${PYTHON_VERSION} CACHE STRING "")
#	SET(PYTHON_LIBRARIES ${PYTHON}${PYTHON_VERSION}/lib/python${PYTHON_VERSION}/config CACHE STRING "")
#	SET(PYTHON_DEBUG_LIBRARIES ${PYTHON}${PYTHON_VERSION}/lib/python${PYTHON_VERSION}/config CACHE STRING "")
	FIND_PACKAGE(PythonLibs)
	INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})

	# boost
	# SET(BOOST_INC ${BOOST_PATH})
	# INCLUDE_DIRECTORIES(${BOOST_INC})
	# SET(BOOST_LIBRARY ${BOOST_PATH}/libs/python/build/bin-stage/)
	# SET(BOOSTPYTHON_LIBRARY ${BOOST_LIBRARY}/boost_python)
	
	# osx internal
	find_library(APPLE_CARBON Carbon) 
	
ELSE(APPLE)
	#  windows/unix
	
	# python
	FIND_PACKAGE(PythonLibs)
	INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
	
	# boost
	# SET(BOOST_INC ${BOOST_PATH})
	# INCLUDE_DIRECTORIES(${BOOST_INC})
	# SET(BOOST_LIBRARY ${BOOST_PATH}/libs/python/build/bin-stage/)
	# SET(BOOSTPYTHON_LIBRARY ${BOOST_LIBRARY}/boost_python)
	
ENDIF(APPLE)


# includes 
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(../)

# source
SET(PYEO_SRCS
../eoFunctorStore.cpp 
PyEO.cpp abstract1.cpp algos.cpp random_numbers.cpp geneticOps.cpp selectOne.cpp continuators.cpp
reduce.cpp replacement.cpp selectors.cpp breeders.cpp mergers.cpp valueParam.cpp 
perf2worth.cpp monitors.cpp statistics.cpp
)

# shared library
ADD_LIBRARY(PyEO MODULE ${PYEO_SRCS})

# python 2.5 must have pyd
IF(WIN32 AND NOT CYGWIN)
	SET_TARGET_PROPERTIES(PyEO PROPERTIES SUFFIX ".pyd")
ENDIF(WIN32 AND NOT CYGWIN)

# add the libs
if(APPLE)
	target_link_libraries(PyEO ${APPLE_CARBON} ${PYTHON_LIBRARY}  ${Boost_LIBRARIES}
	  #${BOOSTPYTHON_LIBRARY}
	  )
else(APPLE)
	target_link_libraries(PyEO ${PYTHON_LIBRARY}  ${Boost_LIBRARIES}
	  #${BOOSTPYTHON_LIBRARY}
	  )
endif(APPLE)
