######################################################################################
### 0) Include the sources
######################################################################################

include_directories(${EO_SRC_DIR}/src)
include_directories(${MOEO_SRC_DIR}/src)

######################################################################################
### 2) Define your target(s): just the peo library here
######################################################################################

set(CORE_LIB_OUTPUT_PATH ${PEO_BIN_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${CORE_LIB_OUTPUT_PATH})

set(CORE_SOURCES 	peo_init.cpp
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
					ring_topo.cpp
					star_topo.cpp
					random_topo.cpp
					complete_topo.cpp)

add_library(peo STATIC ${CORE_SOURCES})

install(TARGETS peo ARCHIVE DESTINATION local/${LIB} COMPONENT libraries)
