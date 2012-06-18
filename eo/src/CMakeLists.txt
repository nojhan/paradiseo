######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Optionnal: add MPI and Boost MPI dependencies.
######################################################################################

IF(WITH_MPI)
    MESSAGE("[EO] Compilation with MPI and BoostMPI.")

    SET(CMAKE_CXX_COMPILER "${MPI_DIR}/bin/mpicxx")

    # headers location
    INCLUDE_DIRECTORIES(${MPI_DIR}/include)
    INCLUDE_DIRECTORIES(${BOOST_DIR}/include)

    # lib location
    LINK_DIRECTORIES(${MPI_DIR}/lib)
    LINK_DIRECTORIES(${BOOST_DIR}/lib)

    # for conditional compilation in code
    ADD_DEFINITIONS(-DWITH_MPI)

    LINK_LIBRARIES(boost_mpi boost_serialization)
ENDIF()

######################################################################################
### 3) Define the eo target
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
  eompi.cpp
  )

ADD_LIBRARY(eo STATIC ${EO_SOURCES})
INSTALL(TARGETS eo ARCHIVE DESTINATION lib COMPONENT libraries)

FILE(GLOB HDRS *.h eo)
INSTALL(FILES ${HDRS} DESTINATION include/eo COMPONENT headers)

######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(EO_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(eo PROPERTIES VERSION "${EO_VERSION}")

######################################################################################
### 5) Where must cmake go now ?
######################################################################################

ADD_SUBDIRECTORY(do)
ADD_SUBDIRECTORY(es)
ADD_SUBDIRECTORY(ga)
ADD_SUBDIRECTORY(gp)
ADD_SUBDIRECTORY(other)
ADD_SUBDIRECTORY(utils)
ADD_SUBDIRECTORY(serial)

IF(ENABLE_PYEO)
  ADD_SUBDIRECTORY(pyeo)
ENDIF(ENABLE_PYEO)

######################################################################################
