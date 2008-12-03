SET(CMAKE_CXX_FLAGS_RELEASE "-O0 -g")

######################################################################################
### 0) Set the compiler
######################################################################################

SET (CMAKE_CXX_COMPILER mpicxx)

######################################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${XML2_CFLAGS}")

######################################################################################


######################################################################################
### 2) Define your target(s): just the rmc_mpi library here
######################################################################################

SET(RMC_MPI_LIB_OUTPUT_PATH ${ParadisEO-PEO_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${RMC_MPI_LIB_OUTPUT_PATH})

SET (RMC_MPI_SOURCES node.cpp 
					param.cpp 
					comm.cpp 
					cooperative.cpp 
					mess.cpp 
					rmc.cpp 
					scheduler.cpp
					synchron.cpp
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

SET(RMC_MPI_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(rmc_mpi PROPERTIES VERSION "${RMC_MPI_VERSION}")
######################################################################################

