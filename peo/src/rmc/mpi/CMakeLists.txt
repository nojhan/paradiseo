set(CMAKE_CXX_FLAGS_RELEASE "-O0 -g")

######################################################################################
### 0) set the compiler
######################################################################################

set(CMAKE_CXX_COMPILER mpicxx)

######################################################################################
### 1) Include the sources
######################################################################################

include_directories(${EO_SRC_DIR}/src)
include_directories(${LIBXML2_INCLUDE_DIR})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

######################################################################################
### 2) Define your target(s): just the rmc_mpi library here
######################################################################################

set(RMC_MPI_LIB_OUTPUT_PATH ${PEO_BIN_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${RMC_MPI_LIB_OUTPUT_PATH})

set (RMC_MPI_SOURCES node.cpp 
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
						
add_library(rmc_mpi STATIC ${RMC_MPI_SOURCES})
add_dependencies(rmc_mpi peo)
install(TARGETS rmc_mpi ARCHIVE DESTINATION ${LIB} COMPONENT libraries)
