INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src
					${MO_SRC_DIR}/src
					${PROBLEMS_SRC_DIR}
					${CMAKE_CURRENT_SOURCE_DIR}/../src)
					
LINK_DIRECTORIES(${EO_BIN_DIR}/lib)	

ADD_EXECUTABLE(lesson1_simpleHC lesson1_simpleHC.cpp) 
ADD_EXECUTABLE(lesson1_firstImprHC lesson1_firstImprHC.cpp) 
ADD_EXECUTABLE(lesson1_randomBestHC lesson1_randomBestHC.cpp) 
ADD_EXECUTABLE(lesson1_neutralHC lesson1_neutralHC.cpp) 
ADD_EXECUTABLE(lesson1_iterContinuator lesson1_iterContinuator.cpp) 
ADD_EXECUTABLE(lesson1_fitContinuator lesson1_fitContinuator.cpp) 
ADD_EXECUTABLE(lesson1_fullEvalContinuator lesson1_fullEvalContinuator.cpp) 
ADD_EXECUTABLE(lesson1_evalContinuator lesson1_evalContinuator.cpp) 
ADD_EXECUTABLE(lesson1_combinedContinuator lesson1_combinedContinuator.cpp) 
ADD_EXECUTABLE(firstImprHC_maxSAT firstImprHC_maxSAT.cpp) 

TARGET_LINK_LIBRARIES(lesson1_simpleHC eoutils ga eo) 
TARGET_LINK_LIBRARIES(lesson1_firstImprHC eoutils ga eo) 
TARGET_LINK_LIBRARIES(lesson1_randomBestHC eoutils ga eo) 
TARGET_LINK_LIBRARIES(lesson1_neutralHC eoutils ga eo) 
TARGET_LINK_LIBRARIES(lesson1_iterContinuator eoutils ga eo) 
TARGET_LINK_LIBRARIES(lesson1_fitContinuator eoutils ga eo) 
TARGET_LINK_LIBRARIES(lesson1_fullEvalContinuator eoutils ga eo) 
TARGET_LINK_LIBRARIES(lesson1_evalContinuator eoutils ga eo) 
TARGET_LINK_LIBRARIES(lesson1_combinedContinuator eoutils ga eo) 
TARGET_LINK_LIBRARIES(firstImprHC_maxSAT eoutils ga eo) 
