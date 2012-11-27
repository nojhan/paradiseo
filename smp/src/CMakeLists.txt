######################################################################################
### 0) Include directories
######################################################################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 1) Set paths
######################################################################################

set(SMP_LIB_OUTPUT_PATH ${SMP_BIN_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${SMP_LIB_OUTPUT_PATH})

######################################################################################
### 2) Build lib
######################################################################################

set (SMP_FILE 
    thread.cpp
    MWModel.h
    scheduler.h
    island.h
    topology/abstractTopology.h
    topology/topology.h
    topology/topologyBuilder.h
    topology/complete.cpp
    topology/star.cpp
    topology/ring.cpp
    topology/hypercubic.cpp
    )

add_library(smp STATIC ${SMP_FILE})

install(TARGETS smp ARCHIVE DESTINATION ${LIB} COMPONENT libraries)

######################################################################################
### 3) Look for headers
######################################################################################

file(GLOB HDRS smp *.h)
install(FILES ${HDRS} DESTINATION include${INSTALL_SUB_DIR}/smp COMPONENT headers)

######################################################################################
### 4) Install directories
######################################################################################

install(DIRECTORY MWAlgo
        DESTINATION include${INSTALL_SUB_DIR}/smp
        COMPONENT headers
        FILES_MATCHING PATTERN "*.h"
        )
