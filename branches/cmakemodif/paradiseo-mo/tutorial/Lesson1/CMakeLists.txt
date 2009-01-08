######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src/utils)
INCLUDE_DIRECTORIES(${MO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${FUNCTION_SRC_DIR})

######################################################################################

######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${FUNCTION_BIN_DIR}/lib)

######################################################################################


######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

IF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008")
	SOURCE_GROUP(src FILES minimize_function.cpp)	
	ADD_EXECUTABLE(minimize_function 
		minimize_function.cpp
		${MO_BIN_DIR}/tutorial/Lesson1/param
	) 
ELSE(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008")
	ADD_COMMANDS_MO()
	ADD_TARGET_MO(lesson1)
	IF(ENABLE_CMAKE_EXAMPLE)
		ADD_EXECUTABLE(minimize_function minimize_function.cpp)
	ELSE(ENABLE_CMAKE_EXAMPLE)	
		ADD_EXECUTABLE(minimize_function EXCLUDE_FROM_ALL minimize_function.cpp)
	ENDIF(ENABLE_CMAKE_EXAMPLE) 		
ENDIF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008")

ADD_DEPENDENCIES(minimize_function function)

######################################################################################





######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################


SET(MINIMIZEFUNCTION_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(minimize_function PROPERTIES VERSION "${MINIMIZEFUNCTION_VERSION}")
######################################################################################



######################################################################################
### 5) Link the librairies for your target(s)
######################################################################################

TARGET_LINK_LIBRARIES(minimize_function function eo eoutils) 

######################################################################################
