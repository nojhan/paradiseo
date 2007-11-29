SET(CMAKE_DEFAULT_BUILD_TYPE Debug CACHE STRING "Variable that stores the default CMake build type" FORCE)
SET(CMAKE_CXX_FLAGS "-O0 -g")
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
SET(CMAKE_CXX_FLAGS_RELEASE "-O0 -g")
SET(CMAKE_CXX_FLAGS_MINSIZEREL "-O0 -g")
SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O0 -g")

######################################################################################
### 0) Set the compiler
######################################################################################

SET (CMAKE_CXX_COMPILER mpicxx)

######################################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${XML2_CFLAGS_WITH_WHITESPACE})

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


######################################################################################
### 4) Link the librairies for your target(s)
######################################################################################

# No library required

######################################################################################


######################################################################################
### 5) Windows advanced config - especially for Microsoft Visual Studio 8
######################################################################################

  IF(CMAKE_CXX_COMPILER MATCHES cl)
   IF(NOT WITH_SHARED_LIBS)
     IF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005")
       SET(CMAKE_CXX_FLAGS "/nologo /W3 /Gy")
       SET(CMAKE_CXX_FLAGS_DEBUG "/MTd /Z7 /Od")
       SET(CMAKE_CXX_FLAGS_RELEASE "/MT /O2")
       SET(CMAKE_CXX_FLAGS_MINSIZEREL "/MT /O2")
       SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MTd /Z7 /Od")
       SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")
       
     ENDIF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005")
   ENDIF(NOT WITH_SHARED_LIBS)
  ENDIF(CMAKE_CXX_COMPILER MATCHES cl)
######################################################################################


######################################################################################
### 6) Where must cmake go now ?
######################################################################################

# nothing to be compiled in the subdirs

######################################################################################

