INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src
					${MO_SRC_DIR}/src
					${MOEO_SRC_DIR}/src
					${PROBLEMS_SRC_DIR}
					${CMAKE_CURRENT_SOURCE_DIR}/../src)
					
LINK_DIRECTORIES(${EO_BIN_DIR}/lib)	

ADD_EXECUTABLE(testPopLS testPopLS.cpp) 

TARGET_LINK_LIBRARIES(testPopLS eoutils ga eo) 
