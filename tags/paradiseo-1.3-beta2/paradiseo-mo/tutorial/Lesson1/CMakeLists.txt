######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src
					${MO_SRC_DIR}/src
					${PROBLEMS_SRC_DIR}
					${CMAKE_CURRENT_SOURCE_DIR}/../src)
	
######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################	
					
LINK_DIRECTORIES(${EO_BIN_DIR}/lib)	


######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

IF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")
	SOURCE_GROUP(src FILES lesson1_simpleHC.cpp
							lesson1_firstImprHC.cpp
							lesson1_randomBestHC.cpp
							lesson1_neutralHC.cpp
							lesson1_iterContinuator.cpp
							lesson1_fitContinuator.cpp
							lesson1_fullEvalContinuator.cpp
							lesson1_evalContinuator.cpp
							lesson1_combinedContinuator.cpp
							firstImprHC_maxSAT.cpp)
	ADD_EXECUTABLE(lesson1_simpleHC 
		lesson1_simpleHC.cpp
		${MO_BIN_DIR}/tutorial/lesson1/lesson1_simpleHC.param
	)
	ADD_EXECUTABLE(lesson1_firstImprHC
		lesson1_firstImprHC.cpp
		${MO_BIN_DIR}/tutorial/lesson1/lesson1_firstImprHC.param
	)
	ADD_EXECUTABLE(lesson1_randomBestHC 
		lesson1_randomBestHC.cpp
		${MO_BIN_DIR}/tutorial/lesson1/lesson1_randomBestHC.param
	)
	ADD_EXECUTABLE(lesson1_neutralHC 
		lesson1_neutralHC.cpp
		${MO_BIN_DIR}/tutorial/lesson1/lesson1_neutralHC.param
	)
	ADD_EXECUTABLE(lesson1_iterContinuator
		lesson1_iterContinuator.cpp
		${MO_BIN_DIR}/tutorial/lesson1/lesson1_iterContinuator.param
	)
	ADD_EXECUTABLE(lesson1_fitContinuator 
		lesson1_fitContinuator.cpp
		${MO_BIN_DIR}/tutorial/lesson1/lesson1_fitContinuator.param
	)
	ADD_EXECUTABLE(lesson1_fullEvalContinuator 
		lesson1_fullEvalContinuator.cpp
		${MO_BIN_DIR}/tutorial/lesson1/lesson1_fullEvalContinuator.param
	)
	ADD_EXECUTABLE(lesson1_evalContinuator 
		lesson1_evalContinuator.cpp
		${MO_BIN_DIR}/tutorial/lesson1/lesson1_evalContinuator.param
	)
	ADD_EXECUTABLE(lesson1_combinedContinuator 
		lesson1_combinedContinuator.cpp
		${MO_BIN_DIR}/tutorial/lesson1/lesson1_combinedContinuator.param
	)
	ADD_EXECUTABLE(firstImprHC_maxSAT
		firstImprHC_maxSAT.cpp
		${MO_BIN_DIR}/tutorial/lesson1/firstImprHC_maxSAT.param
	)
ELSE(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")	
	ADD_COMMANDS_NEWMO()
	ADD_TARGET_NEWMO(lesson1)
	IF(ENABLE_CMAKE_EXAMPLE)
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
	ELSE(ENABLE_CMAKE_EXAMPLE)
		ADD_EXECUTABLE(lesson1_simpleHC EXCLUDE_FROM_ALL lesson1_simpleHC.cpp) 
		ADD_EXECUTABLE(lesson1_firstImprHC EXCLUDE_FROM_ALL lesson1_firstImprHC.cpp) 
		ADD_EXECUTABLE(lesson1_randomBestHC EXCLUDE_FROM_ALL lesson1_randomBestHC.cpp) 
		ADD_EXECUTABLE(lesson1_neutralHC EXCLUDE_FROM_ALL lesson1_neutralHC.cpp) 
		ADD_EXECUTABLE(lesson1_iterContinuator EXCLUDE_FROM_ALL lesson1_iterContinuator.cpp) 
		ADD_EXECUTABLE(lesson1_fitContinuator EXCLUDE_FROM_ALL lesson1_fitContinuator.cpp) 
		ADD_EXECUTABLE(lesson1_fullEvalContinuator EXCLUDE_FROM_ALL lesson1_fullEvalContinuator.cpp) 
		ADD_EXECUTABLE(lesson1_evalContinuator EXCLUDE_FROM_ALL lesson1_evalContinuator.cpp) 
		ADD_EXECUTABLE(lesson1_combinedContinuator EXCLUDE_FROM_ALL lesson1_combinedContinuator.cpp) 
		ADD_EXECUTABLE(firstImprHC_maxSAT EXCLUDE_FROM_ALL firstImprHC_maxSAT.cpp) 
	ENDIF(ENABLE_CMAKE_EXAMPLE)
ENDIF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")

######################################################################################
### 4) Link the librairies for your target(s)
######################################################################################

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
