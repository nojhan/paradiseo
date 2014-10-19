######################################################################################
### 1) Include the sources
######################################################################################

#include_directories(${EO_SRC_DIR}/src)
#include_directories(${EOSERIAL_SRC_DIR}/src)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR})

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

install(FILES eompi.h DESTINATION ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG} COMPONENT headers)

#file(GLOB HDRS *.h)
set(HDRS 
	implMpi.h
	eoMpiNode.h
	eoParallelApply.h
	eoMpi.h
	eoMultiStart.h
	eoTerminateJob.h
	eoMpiAssignmentAlgorithm.h
)

install(FILES ${HDRS} DESTINATION ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG}/eompi COMPONENT headers)

######################################################################################
### 3) Optionnal
######################################################################################

set(EOMPI_VERSION ${GLOBAL_VERSION})
set_target_properties(eompi PROPERTIES VERSION "${EOMPI_VERSION}")

######################################################################################
