
######################################################################################
### 0) Copy the "benchs" directory in the build directory to easily run the lessons
######################################################################################

ADD_CUSTOM_TARGET(install DEPENDS ${ParadisEO-PEO_SOURCE_DIR}/tutorial/examples/tsp/benchs)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_directory 
      ${ParadisEO-PEO_SOURCE_DIR}/tutorial/examples/tsp/benchs
      ${ParadisEO-PEO_BINARY_DIR}/tutorial/examples/tsp/benchs)

######################################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR})
INCLUDE_DIRECTORIES(${MO_SRC_DIR})
INCLUDE_DIRECTORIES(${PEO_SRC_DIR})

######################################################################################


######################################################################################
### 2) Define your target(s): just the tsp lib here
######################################################################################

SET(TSP_LIB_OUTPUT_PATH ${TSP_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${TSP_LIB_OUTPUT_PATH})

SET (TSP_SOURCES 	data.h 
					data.cpp 
					opt_route.h 
					opt_route.cpp 
					param.h 
					param.cpp 
					node.h 
					node.cpp 
					route.h 
					route.cpp 
					route_init.h 
					route_init.cpp 
					route_eval.h 
					route_eval.cpp 
					two_opt.h 
					two_opt.cpp 
					two_opt_init.h 
					two_opt_init.cpp 
					two_opt_incr_eval.h 
					two_opt_incr_eval.cpp 
					two_opt_next.h 
					two_opt_next.cpp 
					order_xover.h 
					order_xover.cpp 
					partial_mapped_xover.h 
					partial_mapped_xover.cpp 
					edge_xover.h 
					edge_xover.cpp 
					city_swap.h 
					city_swap.cpp 
					part_route_eval.h 
					part_route_eval.cpp 
					merge_route_eval.h 
					merge_route_eval.cpp)

ADD_LIBRARY(tsp STATIC ${TSP_SOURCES})  		
######################################################################################


######################################################################################
### 3) Optionnal: define your lib's version
######################################################################################

SET(TSP_VERSION "1.0.beta")
SET_TARGET_PROPERTIES(tsp PROPERTIES VERSION "${TSP_VERSION}")
######################################################################################

