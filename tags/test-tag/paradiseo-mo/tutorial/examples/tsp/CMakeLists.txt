######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${MO_SRC_DIR}/src)

######################################################################################


######################################################################################
### 2) Define your target(s): just the tsp here
######################################################################################

SET(TSP_LIB_OUTPUT_PATH ${TSP_BIN_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${TSP_LIB_OUTPUT_PATH})

SET (TSP_SOURCES graph.cpp 
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
					
ADD_LIBRARY(tsp STATIC ${TSP_SOURCES})

######################################################################################


######################################################################################
### 3) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(TSP_VERSION "${GLOBAL_VERSION}")
SET_TARGET_PROPERTIES(tsp PROPERTIES VERSION "${TSP_VERSION}")
######################################################################################
