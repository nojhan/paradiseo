

######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR})

######################################################################################


######################################################################################
### 2) Define your target(s): just the peo library here
######################################################################################

SET(CORE_LIB_OUTPUT_PATH ${PEO_DIR}/build)
SET(LIBRARY_OUTPUT_PATH ${CORE_LIB_OUTPUT_PATH})

SET (CORE_SOURCES 	peo_init.cpp
					peo_fin.cpp 
					peo_run.cpp 
					peo_param.cpp 
					peo_debug.cpp 
					thread.cpp 
					reac_thread.cpp 
					service.cpp 
					runner.cpp 
					communicable.cpp 
					topology.cpp 
					ring_topo.cpp)



ADD_LIBRARY(peo STATIC ${CORE_SOURCES})

######################################################################################



######################################################################################
### 3) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(CORE_VERSION "1.0.beta")
SET_TARGET_PROPERTIES(peo PROPERTIES VERSION "${CORE_VERSION}")
######################################################################################



