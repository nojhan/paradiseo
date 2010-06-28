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
	SOURCE_GROUP(src FILES testRandomWalk.cpp
							testMetropolisHasting.cpp
							testRandomNeutralWalk.cpp
							sampling.cpp
							densityOfStates.cpp
							autocorrelation.cpp
							adaptiveWalks.cpp
							fdc.cpp
							neutralDegree.cpp
							fitnessCloud.cpp
							neutralWalk.cpp)
	ADD_EXECUTABLE(testRandomWalk
		testRandomWalk.cpp
		${MO_BIN_DIR}/tutorial/lesson6/testRandomWalk.param
	)
	ADD_EXECUTABLE(testMetropolisHasting
		testMetropolisHasting.cpp
		${MO_BIN_DIR}/tutorial/lesson6/testMetropolisHasting.param
	)
	ADD_EXECUTABLE(testRandomNeutralWalk
		testRandomNeutralWalk.cpp
		${MO_BIN_DIR}/tutorial/lesson6/testRandomNeutralWalk.param
	)
	ADD_EXECUTABLE(sampling
		sampling.cpp
		${MO_BIN_DIR}/tutorial/lesson6/sampling.param
	)
	ADD_EXECUTABLE(densityOfStates
		densityOfStates.cpp
		${MO_BIN_DIR}/tutorial/lesson6/densityOfStates.param
	)
	ADD_EXECUTABLE(autocorrelation
		autocorrelation.cpp
		${MO_BIN_DIR}/tutorial/lesson6/autocorrelation.param
	)
	ADD_EXECUTABLE(adaptiveWalks
		adaptiveWalks.cpp
		${MO_BIN_DIR}/tutorial/lesson6/adaptiveWalks.param
	)
	ADD_EXECUTABLE(fdc
		fdc.cpp
		${MO_BIN_DIR}/tutorial/lesson6/fdc.param
	)
	ADD_EXECUTABLE(neutralDegree
		neutralDegree.cpp
		${MO_BIN_DIR}/tutorial/lesson6/neutralDegree.param
	)
	ADD_EXECUTABLE(fitnessCloud
		fitnessCloud.cpp
		${MO_BIN_DIR}/tutorial/lesson6/fitnessCloud.param
	)
	ADD_EXECUTABLE(neutralWalk
		neutralWalk.cpp
		${MO_BIN_DIR}/tutorial/lesson6/neutralWalk.param
	)

ELSE(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")	
	ADD_COMMANDS_NEWMO()
	ADD_TARGET_NEWMO(lesson6)
	IF(ENABLE_CMAKE_EXAMPLE)
		ADD_EXECUTABLE(testRandomWalk testRandomWalk.cpp) 
		ADD_EXECUTABLE(testMetropolisHasting testMetropolisHasting.cpp) 
		ADD_EXECUTABLE(testRandomNeutralWalk testRandomNeutralWalk.cpp) 
		ADD_EXECUTABLE(sampling sampling.cpp) 
		ADD_EXECUTABLE(densityOfStates densityOfStates.cpp) 
		ADD_EXECUTABLE(autocorrelation autocorrelation.cpp) 
		ADD_EXECUTABLE(adaptiveWalks adaptiveWalks.cpp) 
		ADD_EXECUTABLE(fdc fdc.cpp) 
		ADD_EXECUTABLE(neutralDegree neutralDegree.cpp) 
		ADD_EXECUTABLE(fitnessCloud fitnessCloud.cpp) 
		ADD_EXECUTABLE(neutralWalk neutralWalk.cpp) 
	ELSE(ENABLE_CMAKE_EXAMPLE)
		ADD_EXECUTABLE(testRandomWalk EXCLUDE_FROM_ALL testRandomWalk.cpp) 
		ADD_EXECUTABLE(testMetropolisHasting EXCLUDE_FROM_ALL testMetropolisHasting.cpp) 
		ADD_EXECUTABLE(testRandomNeutralWalk EXCLUDE_FROM_ALL testRandomNeutralWalk.cpp) 
		ADD_EXECUTABLE(sampling EXCLUDE_FROM_ALL sampling.cpp) 
		ADD_EXECUTABLE(densityOfStates EXCLUDE_FROM_ALL densityOfStates.cpp) 
		ADD_EXECUTABLE(autocorrelation EXCLUDE_FROM_ALL autocorrelation.cpp) 
		ADD_EXECUTABLE(adaptiveWalks EXCLUDE_FROM_ALL adaptiveWalks.cpp) 
		ADD_EXECUTABLE(fdc EXCLUDE_FROM_ALL fdc.cpp) 
		ADD_EXECUTABLE(neutralDegree EXCLUDE_FROM_ALL neutralDegree.cpp) 
		ADD_EXECUTABLE(fitnessCloud EXCLUDE_FROM_ALL fitnessCloud.cpp) 
		ADD_EXECUTABLE(neutralWalk EXCLUDE_FROM_ALL neutralWalk.cpp) 
	ENDIF(ENABLE_CMAKE_EXAMPLE)
ENDIF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")

######################################################################################
### 4) Link the librairies for your target(s)
######################################################################################

TARGET_LINK_LIBRARIES(testRandomWalk eoutils ga eo) 
TARGET_LINK_LIBRARIES(testMetropolisHasting eoutils ga eo) 
TARGET_LINK_LIBRARIES(testRandomNeutralWalk eoutils ga eo)
TARGET_LINK_LIBRARIES(sampling eoutils ga eo)
TARGET_LINK_LIBRARIES(densityOfStates eoutils ga eo)
TARGET_LINK_LIBRARIES(autocorrelation eoutils ga eo)
TARGET_LINK_LIBRARIES(adaptiveWalks eoutils ga eo)
TARGET_LINK_LIBRARIES(fdc eoutils ga eo)
TARGET_LINK_LIBRARIES(neutralDegree eoutils ga eo)
TARGET_LINK_LIBRARIES(fitnessCloud eoutils ga eo)
TARGET_LINK_LIBRARIES(neutralWalk eoutils ga eo)
