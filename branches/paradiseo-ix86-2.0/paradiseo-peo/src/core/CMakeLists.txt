
######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${MOEO_SRC_DIR}/src)
######################################################################################


######################################################################################
### 2) Define your target(s): just the peo library here
######################################################################################

SET(CORE_LIB_OUTPUT_PATH ${ParadisEO-PEO_BINARY_DIR}/lib)
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

SET(CORE_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(peo PROPERTIES VERSION "${CORE_VERSION}")
######################################################################################


######################################################################################
### 4) Windows advanced config - especially for Microsoft Visual Studio 8
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
### 5) Where must cmake go now ?
######################################################################################

# nothing to be compiled in the subdirs

######################################################################################




