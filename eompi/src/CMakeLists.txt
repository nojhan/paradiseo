######################################################################################
### 1) Include the sources
######################################################################################

include_directories(${EO_SRC_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Define the eompi target
######################################################################################

set(EOMPI_LIB_OUTPUT_PATH ${EOMPI_BIN_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${EOMPI_LIB_OUTPUT_PATH})

set(EOMPI_SOURCES
    eoMpi.cpp
    eoMpiAssignmentAlgorithm.cpp
    eoMpiNode.cpp
    implMpi.cpp
    )

add_library(eompi STATIC ${EOMPI_SOURCES})
install(TARGETS eompi ARCHIVE DESTINATION ${LIB} COMPONENT libraries)

file(GLOB HDRS *.h)
install(FILES ${HDRS} DESTINATION include${INSTALL_SUB_DIR}/eompi COMPONENT headers)

######################################################################################
### 3) Optionnal
######################################################################################

set(EOMPI_VERSION ${GLOBAL_VERSION})
set_target_properties(eompi PROPERTIES VERSION "${EOMPI_VERSION}")

######################################################################################
