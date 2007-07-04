
######################################################################################
### 0) Set the compiler
######################################################################################

SET (CMAKE_CXX_COMPILER mpicxx)

######################################################################################



######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR})
INCLUDE_DIRECTORIES(${XML2_CFLAGS_WITH_WHITESPACE})

######################################################################################



######################################################################################
### 2) Define your target(s): just the core library here
######################################################################################

SET(RMC_MPI_LIB_OUTPUT_PATH ${PEO_DIR}/build)
SET(LIBRARY_OUTPUT_PATH ${RMC_MPI_LIB_OUTPUT_PATH})

SET (RMC_MPI_SOURCES node.cpp 
					param.cpp 
					comm.cpp 
					coop.cpp 
					mess.cpp 
					rmc.cpp 
					scheduler.cpp
					worker.cpp 
					send.cpp 
					recv.cpp 
					xml_parser.cpp 
					schema.cpp 
					runner.cpp 
					service.cpp)
						
ADD_LIBRARY(rmc_mpi STATIC ${RMC_MPI_SOURCES})
ADD_DEPENDENCIES(rmc_mpi peo)
######################################################################################



######################################################################################
### 3) Optionnal: define your lib version: 
######################################################################################

SET(RMC_MPI_VERSION "1.0.beta")
SET_TARGET_PROPERTIES(rmc_mpi PROPERTIES VERSION "${RMC_MPI_VERSION}")
######################################################################################



