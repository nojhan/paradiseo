
######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${EO_SRC_DIR}/utils)
INCLUDE_DIRECTORIES(${ParadisEO-MOEO_SOURCE_DIR}/src)
######################################################################################



######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################
 
LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${ParadisEO-MOEO_BINARY_DIR}/lib)

######################################################################################



######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################
IF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008")
	SOURCE_GROUP(src FILES Sch1.cpp)
	ADD_EXECUTABLE(Sch1 
		Sch1.cpp
		${ParadisEO-MOEO_BINARY_DIR}/tutorial/Lesson1/Sch1.param
	)
ELSE(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008")	
	ADD_COMMANDS_MOEO()
	ADD_TARGET_MOEO(lesson1)
	IF(ENABLE_CMAKE_EXAMPLE)
		ADD_EXECUTABLE(Sch1 Sch1.cpp)
	ELSE(ENABLE_CMAKE_EXAMPLE)
		ADD_EXECUTABLE(Sch1 EXCLUDE_FROM_ALL Sch1.cpp)
	ENDIF(ENABLE_CMAKE_EXAMPLE)
ENDIF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008")

ADD_DEPENDENCIES(Sch1 moeo)
######################################################################################



######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(SCH1_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(Sch1 PROPERTIES VERSION "${SCH1_VERSION}")
######################################################################################



######################################################################################
### 5) Link the librairies for your target(s)
######################################################################################


TARGET_LINK_LIBRARIES(Sch1 moeo eo eoutils)

######################################################################################
