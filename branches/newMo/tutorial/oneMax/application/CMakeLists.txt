INCLUDE_DIRECTORIES(${PARADISEO_EO_SRC_DIR}/src
					${NEWMO_SRC_DIR}/src
					${CMAKE_CURRENT_SOURCE_DIR}/../src)
					
LINK_DIRECTORIES(${PARADISEO_EO_BIN_DIR}/lib)	

ADD_EXECUTABLE(testSimpleHC testSimpleHC.cpp) 
ADD_EXECUTABLE(testFirstImpr testFirstImpr.cpp) 
ADD_EXECUTABLE(testRandomWalk testRandomWalk.cpp) 
ADD_EXECUTABLE(testSimpleHCneutral testSimpleHCneutral.cpp) 
ADD_EXECUTABLE(testHCneutral testHCneutral.cpp) 
ADD_EXECUTABLE(testMetropolisHasting testMetropolisHasting.cpp) 
#ADD_EXECUTABLE(testWithMove testWithMove.cpp) 

TARGET_LINK_LIBRARIES(testSimpleHC eoutils ga eo) 
TARGET_LINK_LIBRARIES(testFirstImpr eoutils ga eo) 
TARGET_LINK_LIBRARIES(testRandomWalk eoutils ga eo) 
TARGET_LINK_LIBRARIES(testSimpleHCneutral eoutils ga eo) 
TARGET_LINK_LIBRARIES(testHCneutral eoutils ga eo) 
TARGET_LINK_LIBRARIES(testMetropolisHasting eoutils ga eo) 
#TARGET_LINK_LIBRARIES(testWithMove eoutils ga eo) 
