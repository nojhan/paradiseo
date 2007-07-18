
######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR})
INCLUDE_DIRECTORIES(${MOEO_SRC_DIR})

######################################################################################


######################################################################################
### 2) Define your target(s): just  the moeo lib here.
######################################################################################

SET(MOEO_LIB_OUTPUT_PATH ${ParadisEO-MOEO_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${MOEO_LIB_OUTPUT_PATH})

SET (MOEO_SOURCES core/moeoObjectiveVectorTraits.cpp)

# --> UNIX 
IF(UNIX)   
  ADD_LIBRARY(moeo STATIC ${MOEO_SOURCES})
ENDIF(UNIX)
		
# --> WIN
IF(WIN32)  
  ADD_LIBRARY(moeo STATIC ${MOEO_SOURCES})
ENDIF(WIN32)

######################################################################################


######################################################################################
### 3) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(MOEO_VERSION "1.0.beta")
SET_TARGET_PROPERTIES(moeo PROPERTIES VERSION "${MOEO_VERSION}")
######################################################################################


######################################################################################
### 4) Link the librairies for your target(s)
######################################################################################

# No library required

######################################################################################


######################################################################################
### 5) Windows advanced config - especially for Microsoft Visual Studio 8
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


######################################################################################
### 6) Where must cmake go now ?
######################################################################################

# nothing to be compiled in the subdirs

######################################################################################



