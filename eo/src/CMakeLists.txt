
######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################



######################################################################################
### 2) Define the eo target
######################################################################################

SET(EO_LIB_OUTPUT_PATH ${EO_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${EO_LIB_OUTPUT_PATH})

SET (EO_SOURCES     eoFunctorStore.cpp 
					eoPersistent.cpp 
					eoPrintable.cpp 
					eoCtrlCContinue.cpp 
					eoParetoFitness.cpp 
					eoScalarFitnessAssembled.cpp)

 
ADD_LIBRARY(eo STATIC ${EO_SOURCES})

######################################################################################



######################################################################################
### 3) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

SET(EO_VERSION "1.0.1")
SET_TARGET_PROPERTIES(eo PROPERTIES VERSION "${EO_VERSION}")
######################################################################################



######################################################################################
### 4) Windows advanced config - especially for Microsoft Visual Studio 8
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
### 5) Where must cmake go now ?
######################################################################################

SUBDIRS(do es ga gp other utils) # forget pyeo for the moment

######################################################################################

