INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src
					${MO_SRC_DIR}/src
					${PROBLEMS_SRC_DIR}
					${CMAKE_CURRENT_SOURCE_DIR}/../src)
					
LINK_DIRECTORIES(${EO_BIN_DIR}/lib)	


ADD_EXECUTABLE(testRandomWalk testRandomWalk.cpp) 
ADD_EXECUTABLE(testMetropolisHasting testMetropolisHasting.cpp) 
ADD_EXECUTABLE(testRandomNeutralWalk testRandomNeutralWalk.cpp) 

TARGET_LINK_LIBRARIES(testRandomWalk eoutils ga eo) 
TARGET_LINK_LIBRARIES(testMetropolisHasting eoutils ga eo) 
TARGET_LINK_LIBRARIES(testRandomNeutralWalk eoutils ga eo)
