
######################################################################################
### 0) Copy the "benchs" directory in the build directory to easily run the lessons
######################################################################################

ADD_CUSTOM_TARGET(install DEPENDS ${FLOWSHOP_SRC_DIR}/benchs)
ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_directory 
      ${FLOWSHOP_SRC_DIR}/benchs
      ${FLOWSHOP_BINARY_DIR}/benchs)

######################################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR})
INCLUDE_DIRECTORIES(${MOEO_SRC_DIR})
INCLUDE_DIRECTORIES(${FLOWSHOP_SRC_DIR})

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries (mandatory: before 3) )
######################################################################################

# --> UNIX 
IF(UNIX)  
  LINK_DIRECTORIES(${EO_SRC_DIR} ${EO_SRC_DIR}/utils ${ParadisEO-MOEO_BINARY_DIR}/lib)
ENDIF(UNIX)

# --> WIN
IF(WIN32)  
  # "CMAKE_BUILD_TYPE" supposed to be given on the command line, default=Debug
  IF (NOT DEFINED CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Debug)
  ENDIF (NOT DEFINED CMAKE_BUILD_TYPE)
 
  IF (CMAKE_BUILD_TYPE STREQUAL Debug)
  	LINK_DIRECTORIES(${ParadisEO-MOEO_BINARY_DIR}\\lib\\debug)
  	LINK_DIRECTORIES(${EO_LIB_DIR}\\debug)
  ENDIF(CMAKE_BUILD_TYPE STREQUAL Debug)	
  
  IF(CMAKE_BUILD_TYPE STREQUAL Release)
  	LINK_DIRECTORIES(${ParadisEO-MOEO_BINARY_DIR}\\lib\\release)
  	LINK_DIRECTORIES(${EO_LIB_DIR}\\release)  
  ENDIF(CMAKE_BUILD_TYPE STREQUAL Release)
ENDIF(WIN32)  

######################################################################################



######################################################################################
### 3) Define your target(s): just the flowshop lib here
######################################################################################

SET(FLOWSHOP_LIB_OUTPUT_PATH ${FLOWSHOP_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${FLOWSHOP_LIB_OUTPUT_PATH})

SET (FLOWSHOP_SOURCES   FlowShopBenchmarkParser.cpp 
						FlowShopEval.cpp 
						FlowShopInit.cpp 
						FlowShopObjectiveVectorTraits.cpp 
						FlowShopOpCrossoverQuad.cpp 
						FlowShopOpMutationExchange.cpp 
						FlowShopOpMutationShift.cpp 
						FlowShop.cpp)

# --> UNIX 
IF(UNIX)   
  ADD_LIBRARY(flowshop STATIC ${FLOWSHOP_SOURCES})
ENDIF(UNIX)

# --> WIN
IF(WIN32)  
  ADD_LIBRARY(flowshop STATIC ${FLOWSHOP_SOURCES})
ENDIF(WIN32) 

ADD_DEPENDENCIES(flowshop moeo)

ENDIF(UNIX)
######################################################################################



######################################################################################
### 4) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(FLOWSHOP_VERSION "1.0.beta")
SET_TARGET_PROPERTIES(flowshop PROPERTIES VERSION "${FLOWSHOP_VERSION}")
######################################################################################


######################################################################################
### 5) Link the librairies for your target(s)
######################################################################################

# --> UNIX 
IF(UNIX) 
  TARGET_LINK_LIBRARIES(flowshop moeo)
  TARGET_LINK_LIBRARIES(flowshop eo)
ENDIF(UNIX)

# --> APPLE AND WIN
IF(WIN32)  
  # "CMAKE_BUILD_TYPE" supposed to be given on the command line, default=Debug
  IF (NOT DEFINED CMAKE_BUILD_TYPE)
    set (CMAKE_BUILD_TYPE Debug)
  ENDIF (NOT DEFINED CMAKE_BUILD_TYPE)
 
  IF (CMAKE_BUILD_TYPE STREQUAL Debug)
  	TARGET_LINK_LIBRARIES(flowshop moeo)
    TARGET_LINK_LIBRARIES(flowshop eod)
  ENDIF(CMAKE_BUILD_TYPE STREQUAL Debug)
   
  IF(CMAKE_BUILD_TYPE STREQUAL Release)
  	TARGET_LINK_LIBRARIES(flowshop moeo)
    TARGET_LINK_LIBRARIES(flowshop eo)
  ENDIF(CMAKE_BUILD_TYPE STREQUAL Release)
 
ENDIF(WIN32)
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




