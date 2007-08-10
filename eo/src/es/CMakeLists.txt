
######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################


######################################################################################
### 2) Define the es and cma targets
######################################################################################

SET(ES_LIB_OUTPUT_PATH ${EO_BINARY_DIR}/lib)
SET(CMA_LIB_OUTPUT_PATH ${EO_BINARY_DIR}/lib)

SET(LIBRARY_OUTPUT_PATH ${ES_LIB_OUTPUT_PATH}) # the same output for the two libs

SET (ES_SOURCES 	make_algo_scalar_es.cpp
					make_algo_scalar_real.cpp 
					make_checkpoint_es.cpp 
					make_checkpoint_real.cpp 
					make_continue_es.cpp 
					make_continue_real.cpp 
					make_genotype_es.cpp 
					make_genotype_real.cpp 
					make_op_es.cpp 
					make_op_real.cpp 
					make_pop_es.cpp 
					make_pop_real.cpp 
					make_run_es.cpp 
					make_run_real.cpp)
				
SET (CMA_SOURCES 	eig.cpp
					CMAState.cpp
					CMAParams.cpp)

 
ADD_LIBRARY(es STATIC ${ES_SOURCES})
ADD_LIBRARY(cma STATIC ${CMA_SOURCES})

######################################################################################



######################################################################################
### 3) Optionnal
######################################################################################

SET(ES_VERSION "1.0.1")
SET_TARGET_PROPERTIES(es PROPERTIES VERSION "${ES_VERSION}")

SET(CMA_VERSION "1.0.1")
SET_TARGET_PROPERTIES(cma PROPERTIES VERSION "${CMA_VERSION}")
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

