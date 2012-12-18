######################################################################################
### 1) Include the sources
######################################################################################

include_directories(${EO_SRC_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Define the eoserial target
######################################################################################

set(EOSERIAL_LIB_OUTPUT_PATH ${EO_BIN_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${EOSERIAL_LIB_OUTPUT_PATH})

set(EOSERIAL_SOURCES
    Array.cpp
    Object.cpp
    Parser.cpp
    String.cpp
  )

add_library(eoserial STATIC ${EOSERIAL_SOURCES})
install(TARGETS eoserial ARCHIVE DESTINATION lib COMPONENT libraries)

file(GLOB HDRS *.h)
install(FILES ${HDRS} DESTINATION include/eo/serial COMPONENT headers)

######################################################################################
### 3) Optionnal
######################################################################################

set(EOSERIAL_VERSION ${GLOBAL_VERSION})
set_target_properties(eoserial PROPERTIES VERSION "${EOSERIAL_VERSION}")

######################################################################################
