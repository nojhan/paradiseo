INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src
					${MO_SRC_DIR}/src
					${CMAKE_CURRENT_SOURCE_DIR}/../src)
					
LINK_DIRECTORIES(${EO_BIN_DIR}/lib)	

ADD_EXECUTABLE(testSimpleHC testSimpleHC.cpp) 
ADD_EXECUTABLE(testFirstImpr testFirstImpr.cpp) 
ADD_EXECUTABLE(testRandomWalk testRandomWalk.cpp) 
ADD_EXECUTABLE(testSimpleHCneutral testSimpleHCneutral.cpp) 
ADD_EXECUTABLE(testHCneutral testHCneutral.cpp) 
ADD_EXECUTABLE(testMetropolisHasting testMetropolisHasting.cpp) 
#ADD_EXECUTABLE(testWithMove testWithMove.cpp) 
ADD_EXECUTABLE(testSimpleTS testSimpleTS.cpp) 
ADD_EXECUTABLE(testRandomNeutralWalk testRandomNeutralWalk.cpp) 
ADD_EXECUTABLE(testILS testILS.cpp) 
ADD_EXECUTABLE(testSimulatedAnnealing testSimulatedAnnealing.cpp) 

TARGET_LINK_LIBRARIES(testSimpleHC eoutils ga eo) 
TARGET_LINK_LIBRARIES(testFirstImpr eoutils ga eo) 
TARGET_LINK_LIBRARIES(testRandomWalk eoutils ga eo) 
TARGET_LINK_LIBRARIES(testSimpleHCneutral eoutils ga eo) 
TARGET_LINK_LIBRARIES(testHCneutral eoutils ga eo) 
TARGET_LINK_LIBRARIES(testMetropolisHasting eoutils ga eo) 
#TARGET_LINK_LIBRARIES(testWithMove eoutils ga eo) 
TARGET_LINK_LIBRARIES(testSimpleTS eoutils ga eo)
TARGET_LINK_LIBRARIES(testRandomNeutralWalk eoutils ga eo)
TARGET_LINK_LIBRARIES(testILS eoutils ga eo) 
TARGET_LINK_LIBRARIES(testSimulatedAnnealing eoutils ga eo) 

 
