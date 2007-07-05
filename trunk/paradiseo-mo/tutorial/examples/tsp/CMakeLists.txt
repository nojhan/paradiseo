
######################################################################################
### 0) Copy the "benchs" directory in the build directory to easily run the lessons
######################################################################################

ADD_CUSTOM_TARGET(copy_tsp_benchmarks ALL)
ADD_CUSTOM_COMMAND(
    TARGET copy_tsp_benchmarks
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_directory 
      ${ParadisEO-MO_SOURCE_DIR}/tutorial/examples/tsp/benchs
      ${ParadisEO-MO_BINARY_DIR}/tutorial/examples/tsp/benchs)

######################################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR})
INCLUDE_DIRECTORIES(${MO_SRC_DIR})

######################################################################################


######################################################################################
### 2) Define your target(s): just the tsp here
######################################################################################

SET(TSP_LIB_OUTPUT_PATH ${TSP_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${TSP_LIB_OUTPUT_PATH})

SET (TSP_SOURCES 	graph.cpp 
					route_init.cpp  
					route_eval.cpp 
					part_route_eval.cpp 
					edge_xover.cpp 
					order_xover.cpp 
					route_valid.cpp 
					partial_mapped_xover.cpp 
					city_swap.cpp 
					two_opt.cpp	
					two_opt_init.cpp 
					two_opt_next.cpp 
					two_opt_incr_eval.cpp 
					two_opt_tabu_list.cpp 
					two_opt_rand.cpp)

# --> UNIX 
IF(UNIX)   
  ADD_LIBRARY(tsp STATIC ${TSP_SOURCES})
  		
# --> WIN
ELSEIF(WIN32)  
  ADD_LIBRARY(tsp STATIC ${TSP_SOURCES})

ENDIF(UNIX)
######################################################################################



######################################################################################
### 3) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(TSP_VERSION "1.0.beta")
SET_TARGET_PROPERTIES(tsp PROPERTIES VERSION "${TSP_VERSION}")
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

