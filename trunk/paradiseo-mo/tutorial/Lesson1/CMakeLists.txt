INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src
					${MO_SRC_DIR}/src
					${CMAKE_CURRENT_SOURCE_DIR}/../src)
					
LINK_DIRECTORIES(${EO_BIN_DIR}/lib)	

ADD_EXECUTABLE(testSimpleHC testSimpleHC.cpp) 
ADD_EXECUTABLE(testFirstImpr testFirstImpr.cpp) 
ADD_EXECUTABLE(testRandomBestHC testRandomBestHC.cpp) 
ADD_EXECUTABLE(testHCneutral testHCneutral.cpp) 

TARGET_LINK_LIBRARIES(testSimpleHC eoutils ga eo) 
TARGET_LINK_LIBRARIES(testFirstImpr eoutils ga eo) 
TARGET_LINK_LIBRARIES(testRandomBestHC eoutils ga eo) 
TARGET_LINK_LIBRARIES(testHCneutral eoutils ga eo) 
