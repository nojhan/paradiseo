######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Define the eoserial target
######################################################################################

SET(EOSERIAL_LIB_OUTPUT_PATH ${EO_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${EOSERIAL_LIB_OUTPUT_PATH})

SET(EOSERIAL_SOURCES
    Array.cpp
    Object.cpp
    Parser.cpp
    String.cpp
  )

ADD_LIBRARY(eoserial STATIC ${EOSERIAL_SOURCES})
INSTALL(TARGETS eoserial ARCHIVE DESTINATION lib COMPONENT libraries)

FILE(GLOB HDRS *.h)
INSTALL(FILES ${HDRS} DESTINATION include/eo/serial COMPONENT headers)

######################################################################################
### 3) Optionnal
######################################################################################

SET(EOSERIAL_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(eoserial PROPERTIES VERSION "${EOSERIAL_VERSION}")

######################################################################################
