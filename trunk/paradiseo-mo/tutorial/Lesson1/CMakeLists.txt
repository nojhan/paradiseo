INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src
					${MO_SRC_DIR}/src
					${CMAKE_CURRENT_SOURCE_DIR}/../src)
					
LINK_DIRECTORIES(${EO_BIN_DIR}/lib)	

ADD_EXECUTABLE(lesson1_simpleHC lesson1_simpleHC.cpp) 
ADD_EXECUTABLE(lesson1_firstImprHC lesson1_firstImprHC.cpp) 
ADD_EXECUTABLE(lesson1_randomBestHC lesson1_randomBestHC.cpp) 
ADD_EXECUTABLE(lesson1_neutralHC lesson1_neutralHC.cpp) 

TARGET_LINK_LIBRARIES(lesson1_simpleHC eoutils ga eo) 
TARGET_LINK_LIBRARIES(lesson1_firstImprHC eoutils ga eo) 
TARGET_LINK_LIBRARIES(lesson1_randomBestHC eoutils ga eo) 
TARGET_LINK_LIBRARIES(lesson1_neutralHC eoutils ga eo) 
