
######################################################################################
### 0) Copy the "benchs" directory in the build directory
######################################################################################

ADD_CUSTOM_TARGET(install DEPENDS ${CMAKE_SOURCE_DIR}/benchs)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_directory 
      ${CMAKE_SOURCE_DIR}/benchs
      ${CMAKE_BINARY_DIR}/benchs)

##########################################################################################



######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################



######################################################################################
### 2) Define your target(s): just the tsp lib here
######################################################################################

SET(TSP_LIB_OUTPUT_PATH ${CMAKE_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${TSP_LIB_OUTPUT_PATH})

SET (TSP_SOURCES 		
                    city_swap.h 				
					city_swap.cpp   
					edge_xover.h
					edge_xover.cpp  
					graph.h
					graph.cpp                  
                    mix.h
                    order_xover.h       
                    order_xover.cpp
                    part_route_eval.h  
                    part_route_eval.cpp   
                    partial_mapped_xover.h    
                    partial_mapped_xover.cpp                   
                    route.h
                    route_eval.h   
					route_eval.cpp
					route_init.h
                    route_init.cpp  
                    route_valid.h       
                    route_valid.cpp)

ADD_LIBRARY(ltsp STATIC ${TSP_SOURCES})  		
######################################################################################

