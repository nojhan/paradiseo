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

LINK_DIRECTORIES(${EO_BINARY_DIR}/lib)

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
				t-eoInt
				t-eoInitPermutation
				t-eoSwapMutation
				t-eoShiftMutation
				t-eoTwoOptMutation
				t-eoRingTopology
				t-eoSyncEasyPSO
				# t-eoFrontSorter 
				# t-eoEpsMOEA
				)
				
FOREACH (test ${TEST_LIST})
	  SET ("T_${test}_SOURCES" "${test}.cpp")
ENDFOREACH (test)


IF(ENABLE_MINIMAL_CMAKE_TESTING)
    SET (MIN_TEST_LIST t-eoEasyPSO)				
    FOREACH (mintest ${MIN_TEST_LIST})
    	  SET ("T_${mintest}_SOURCES" "${mintest}.cpp")
    	  ADD_EXECUTABLE(${mintest} ${T_${mintest}_SOURCES}) 
		  ADD_TEST(${mintest} ${mintest}) 
		  TARGET_LINK_LIBRARIES(${mintest} ga es cma eoutils eo eomoo)
    ENDFOREACH (mintest)
   
ELSEIF(ENABLE_CMAKE_TESTING)	
	 FOREACH (test ${TEST_LIST})
        ADD_EXECUTABLE(${test} ${T_${test}_SOURCES}) 
        ADD_TEST(${test} ${test}) 
        TARGET_LINK_LIBRARIES(${test} ga es cma eoutils eo eomoo)
    ENDFOREACH (test)
ENDIF(ENABLE_MINIMAL_CMAKE_TESTING)
######################################################################################

