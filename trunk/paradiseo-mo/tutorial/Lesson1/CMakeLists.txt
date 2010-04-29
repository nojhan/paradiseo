INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src
					${MO_SRC_DIR}/src
					${PROBLEMS_SRC_DIR}
					${CMAKE_CURRENT_SOURCE_DIR}/../src)
					
LINK_DIRECTORIES(${EO_BIN_DIR}/lib)	

ADD_EXECUTABLE(lesson1_simpleHC lesson1_simpleHC.cpp) 
ADD_EXECUTABLE(lesson1_firstImprHC lesson1_firstImprHC.cpp) 
ADD_EXECUTABLE(lesson1_randomBestHC lesson1_randomBestHC.cpp) 
ADD_EXECUTABLE(lesson1_neutralHC lesson1_neutralHC.cpp) 
ADD_EXECUTABLE(lesson1_continuator lesson1_continuator.cpp) 

TARGET_LINK_LIBRARIES(lesson1_simpleHC eoutils ga eo) 
TARGET_LINK_LIBRARIES(lesson1_firstImprHC eoutils ga eo) 
TARGET_LINK_LIBRARIES(lesson1_randomBestHC eoutils ga eo) 
TARGET_LINK_LIBRARIES(lesson1_neutralHC eoutils ga eo) 
TARGET_LINK_LIBRARIES(lesson1_continuator eoutils ga eo) 
