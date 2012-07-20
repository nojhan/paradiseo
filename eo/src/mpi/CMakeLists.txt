######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Define the eompi target
######################################################################################

SET(EOMPI_LIB_OUTPUT_PATH ${EO_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${EOMPI_LIB_OUTPUT_PATH})

SET(EOMPI_SOURCES
    eoMpi.cpp
    )

ADD_LIBRARY(eompi STATIC ${EOMPI_SOURCES})
INSTALL(TARGETS eompi ARCHIVE DESTINATION lib COMPONENT libraries)

FILE(GLOB HDRS *.h)
INSTALL(FILES ${HDRS} DESTINATION include/eo/mpi COMPONENT headers)

######################################################################################
### 3) Optionnal
######################################################################################

SET(EOMPI_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(eompi PROPERTIES VERSION "${EOMPI_VERSION}")

######################################################################################
