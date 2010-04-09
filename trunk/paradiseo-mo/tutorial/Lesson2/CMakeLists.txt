INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src
					${MO_SRC_DIR}/src
					${CMAKE_CURRENT_SOURCE_DIR}/../src)
					
LINK_DIRECTORIES(${EO_BIN_DIR}/lib)	

ADD_EXECUTABLE(testNeighborhood testNeighborhood.cpp) 

TARGET_LINK_LIBRARIES(testNeighborhood eoutils ga eo) 
