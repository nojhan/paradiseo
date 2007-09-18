###############################################################################
##
## CMakeLists file for eo/test
##
###############################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/contrib)
INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/contrib/MGE)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

IF(NOT WIN32 OR CYGWIN)  
  LINK_DIRECTORIES(${EO_BINARY_DIR}/lib)
ENDIF(NOT WIN32 OR CYGWIN)

# especially for Visual Studio
IF(WIN32 AND NOT CYGWIN)  
  	LINK_DIRECTORIES(${EO_BINARY_DIR}\\lib\\${CMAKE_BUILD_TYPE})
ENDIF(WIN32 AND NOT CYGWIN) 

######################################################################################



######################################################################################
### 3) Define your targets and link the librairies
######################################################################################

SET (TEST_LIST  t-eoParetoFitness 
				t-eoPareto 
				t-eofitness 
				t-eoRandom 
				t-eobin 
				t-eoVirus 
				t-MGE 
				t-MGE1bit 
				t-MGE-control 
				t-eoStateAndParser 
				t-eoCheckpointing 
				t-eoSSGA 
				t-eoExternalEO 
				t-eoSymreg 
				t-eo 
				t-eoReplacement 
				t-eoSelect 
				t-eoGenOp 
				t-eoGA 
				t-eoReal 
				t-eoVector 
				t-eoESAll 
				t-eoPBIL 
				t-eoFitnessAssembled 
				t-eoFitnessAssembledEA 
				t-eoRoulette 
				t-eoSharing 
				t-eoCMAES 
				t-eoSecondsElapsedContinue
				t-eoRNG		
				t-eoEasyPSO 
				t-eoNSGA 
				# t-eoFrontSorter 
				# t-eoEpsMOEA
				)
				
FOREACH (test ${TEST_LIST})
	  SET ("T_${test}_SOURCES" "${test}.cpp")
ENDFOREACH (test)


IF(ENABLE_CMAKE_TESTING)
	
	# Add the tests
	FOREACH (test ${TEST_LIST})
		  ADD_EXECUTABLE(${test} ${T_${test}_SOURCES}) 
		  ADD_TEST(${test} ${test}) 
	ENDFOREACH (test)
	
	# Link the librairies
	FOREACH (test ${TEST_LIST})
	 TARGET_LINK_LIBRARIES(${test} ga es cma eoutils eo eomoo)
	ENDFOREACH (test)
	
ENDIF(ENABLE_CMAKE_TESTING)

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



