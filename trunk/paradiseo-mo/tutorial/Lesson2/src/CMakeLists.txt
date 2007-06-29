

######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR})
INCLUDE_DIRECTORIES(${EO_SRC_DIR}/utils)
INCLUDE_DIRECTORIES(${MO_SRC_DIR})
INCLUDE_DIRECTORIES(${TSP_EXAMPLE_DIR}/src)
######################################################################################



######################################################################################
### 2) Specify where CMake can find the libraries (mandatory: before 3) )
######################################################################################

# --> UNIX 
IF(UNIX)  
  LINK_DIRECTORIES(${EO_SRC_DIR} ${EO_SRC_DIR}/utils ${TSP_EXAMPLE_DIR}/build)

# --> WIN
ELSEIF(WIN32)  
  # "CMAKE_BUILD_TYPE" supposed to be given on the command line, default=Debug
  IF (NOT DEFINED CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Debug)
  ENDIF (NOT DEFINED CMAKE_BUILD_TYPE)
 
  IF (CMAKE_BUILD_TYPE STREQUAL Debug)
  	LINK_DIRECTORIES(${TSP_EXAMPLE_DIR}\\build\\debug)
  	LINK_DIRECTORIES(${EO_LIB_DIR}\\debug)
  ELSEIF(CMAKE_BUILD_TYPE STREQUAL Release)
  	LINK_DIRECTORIES(${TSP_EXAMPLE_DIR}\\build\\release)
  	LINK_DIRECTORIES(${EO_LIB_DIR}\\release)  
  ENDIF(CMAKE_BUILD_TYPE STREQUAL Debug)

ENDIF(UNIX)

######################################################################################



######################################################################################
### 3) Define your target(s): just an executable here
######################################################################################

# no matter what is the OS, hopefully
ADD_EXECUTABLE(tabu_search tabu_search.cpp) 

ADD_DEPENDENCIES(tabu_search tsp)

SET(EXECUTABLE_OUTPUT_PATH ${TUTORIAL_LESSON2_DIR}/build)
######################################################################################



######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(TABUSEARCH_VERSION "1.0.beta")
SET_TARGET_PROPERTIES(tabu_search PROPERTIES VERSION "${TABUSEARCH_VERSION}")
######################################################################################



######################################################################################
### 5) Link the librairies for your target(s)
######################################################################################

# --> UNIX 
IF(UNIX) 
  TARGET_LINK_LIBRARIES(tabu_search tsp) 
  TARGET_LINK_LIBRARIES(tabu_search eo)
  TARGET_LINK_LIBRARIES(tabu_search eoutils)

# --> WIN
ELSEIF(WIN32)  
  # "CMAKE_BUILD_TYPE" supposed to be given on the command line, default=Debug
  IF (NOT DEFINED CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Debug)
  ENDIF (NOT DEFINED CMAKE_BUILD_TYPE)
 
  IF (CMAKE_BUILD_TYPE STREQUAL Debug)
  	TARGET_LINK_LIBRARIES(tabu_search tsp)
    TARGET_LINK_LIBRARIES(tabu_search eod)
    TARGET_LINK_LIBRARIES(tabu_search eoutilsd)    
  ELSEIF(CMAKE_BUILD_TYPE STREQUAL Release)
  	TARGET_LINK_LIBRARIES(tabu_search tsp)
    TARGET_LINK_LIBRARIES(tabu_search eo)
    TARGET_LINK_LIBRARIES(tabu_search eoutils)  
  ELSE(CMAKE_BUILD_TYPE STREQUAL Debug)
    MESSAGE(FATAL_ERROR " Cmake build type can only be \"Debug\" or \"Release\" ") 
  ENDIF(CMAKE_BUILD_TYPE STREQUAL Debug)

ENDIF(UNIX)
######################################################################################



######################################################################################
### 6) Windows advanced config - especially for Microsoft Visual Studio 8
######################################################################################

  IF(CMAKE_CXX_COMPILER MATCHES cl)
   IF(NOT WITH_SHARED_LIBS)
     IF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005")
       SET(CMAKE_CXX_FLAGS "/nologo /W3 /Gy")
       SET(CMAKE_CXX_FLAGS_DEBUG "/MTd /Z7 /Od")
       SET(CMAKE_CXX_FLAGS_RELEASE "/MT /O2")
       SET(CMAKE_CXX_FLAGS_MINSIZEREL "/MT /O2")
       SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MTd /Z7 /Od")
       SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")
       
     ENDIF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005")
   ENDIF(NOT WITH_SHARED_LIBS)
  ENDIF(CMAKE_CXX_COMPILER MATCHES cl)
######################################################################################



