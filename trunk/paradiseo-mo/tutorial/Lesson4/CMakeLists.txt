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
	SOURCE_GROUP(src FILES testSimpleTS.cpp)
	ADD_EXECUTABLE(testSimpleTS 
		testSimpleTS.cpp
		${MO_BIN_DIR}/tutorial/lesson4/testSimpleTS.param
	)
ELSE(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")	
	ADD_COMMANDS_NEWMO()
	ADD_TARGET_NEWMO(lesson4)
	IF(ENABLE_CMAKE_EXAMPLE)
		ADD_EXECUTABLE(testSimpleTS testSimpleTS.cpp)
	ELSE(ENABLE_CMAKE_EXAMPLE)
		ADD_EXECUTABLE(testSimpleTS EXCLUDE_FROM_ALL testSimpleTS.cpp)
	ENDIF(ENABLE_CMAKE_EXAMPLE)
ENDIF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")

######################################################################################
### 4) Link the librairies for your target(s)
######################################################################################

TARGET_LINK_LIBRARIES(testSimpleTS eoutils ga eo) 

