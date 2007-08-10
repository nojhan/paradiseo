
######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
######################################################################################


######################################################################################
### 2) Define the ga target
######################################################################################

SET(GA_LIB_OUTPUT_PATH ${EO_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${GA_LIB_OUTPUT_PATH})

SET (GA_SOURCES 	make_algo_scalar_ga.cpp
					make_checkpoint_ga.cpp
					make_continue_ga.cpp
					make_genotype_ga.cpp
					make_op_ga.cpp
					make_pop_ga.cpp
					make_run_ga.cpp)


ADD_LIBRARY(ga STATIC ${GA_SOURCES})
######################################################################################



######################################################################################
### 3) Optionnal
######################################################################################

SET(GA_VERSION "1.0.1")
SET_TARGET_PROPERTIES(ga PROPERTIES VERSION "${GA_VERSION}")

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

