######################################################################################
### 1) Include the sources
######################################################################################

include_directories(${EO_SRC_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Define the eoserial target
######################################################################################

set(EOSERIAL_LIB_OUTPUT_PATH ${EOSERIAL_BIN_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${EOSERIAL_LIB_OUTPUT_PATH})

set(EOSERIAL_SOURCES
    SerialArray.cpp
    SerialObject.cpp
    Parser.cpp
    SerialString.cpp
  )

add_library(eoserial STATIC ${EOSERIAL_SOURCES})
install(TARGETS eoserial ARCHIVE DESTINATION ${LIB} COMPONENT libraries)

######################################################################################
### 4) Look for headers
######################################################################################

file(GLOB HDRS *.h)
install(FILES ${HDRS} DESTINATION ${INSTALL_SUB_DIR}/eoserial COMPONENT headers)

######################################################################################
### 5) Optionnal
######################################################################################

set(EOSERIAL_VERSION ${GLOBAL_VERSION})
set_target_properties(eoserial PROPERTIES VERSION "${EOSERIAL_VERSION}")

######################################################################################
