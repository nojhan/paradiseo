######################################################################################
### 0) Include directories
######################################################################################

#include_directories(${EO_SRC_DIR}/src)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 1) Set paths
######################################################################################

set(SMP_LIB_OUTPUT_PATH ${SMP_BIN_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${SMP_LIB_OUTPUT_PATH})

######################################################################################
### 2) Build lib
######################################################################################

set (SMP_FILE 
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
    topology/mesh.cpp
    topology/customBooleanTopology.cpp
    topology/customStochasticTopology.cpp
    notifier.cpp
    islandModelWrapper.h
    )

add_library(smp STATIC ${SMP_FILE})

install(TARGETS smp ARCHIVE DESTINATION ${LIB} COMPONENT libraries)

######################################################################################
### 3) Look for headers
######################################################################################

install(FILES smp.h DESTINATION ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG} COMPONENT headers)

#file(GLOB HDRS smp *.h *.cpp)
set(HDRS
    notifier.h
    abstractIsland.h
    #smp
    contDispatching.h
    island.cpp
    policiesDispatching.h
    MWModel.h
    bimap.cpp
    #smp.h
    contWrapper.cpp
    topology/mesh.cpp
    topology/ring.cpp
    topology/ring.h
    topology/hypercubic.cpp
    topology/complete.h
    topology/abstractTopology.h
    topology/star.cpp
    topology/customBooleanTopology.cpp
    topology/customStochasticTopology.cpp
    topology/topology.cpp
    topology/hypercubic.h
    topology/star.h
    topology/topology.h
    topology/complete.cpp
    topology/mesh.h
    topology/topologyBuilder.h
    topology/customStochasticTopology.h
    topology/customBooleanTopology.h
    contWrapper.h
    PPExpander.h
    bimap.h
    MWModel.cpp
    islandNotifier.cpp
    policyElement.h
    notifier.cpp
    islandModel.h
    islandModelWrapper.h
    scheduler.cpp
    islandNotifier.h
    island.h
    intPolicy.h
    policyElement.cpp
    scheduler.h
    algoDispatching.h
    migPolicy.h
    islandModel.cpp
    MWAlgo/eoEasyEA.cpp
    MWAlgo/eoSyncEasyPSO.cpp
    MWAlgo/eoEasyPSO.cpp
    MWAlgo/MWAlgo.h
)

install(FILES ${HDRS} DESTINATION ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG}/smp COMPONENT headers)

######################################################################################
### 4) Install directories
######################################################################################

install(DIRECTORY MWAlgo topology
        DESTINATION ${PROJECT_HDRS_INSTALL_SUBPATH}/${PROJECT_TAG}/smp
        COMPONENT headers
        FILES_MATCHING PATTERN "*.h"
        PATTERN "*.cpp"
        )
