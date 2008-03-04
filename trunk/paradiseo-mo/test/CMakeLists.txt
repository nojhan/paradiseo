###############################################################################
##
## CMakeLists file for ParadisEO-MO/test
##
###############################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${ParadisEO-MO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

IF(NOT WIN32 OR CYGWIN)  
  LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${ParadisEO-MO_BINARY_DIR}/lib)
ENDIF(NOT WIN32 OR CYGWIN)

# especially for Visual Studio
IF(WIN32 AND NOT CYGWIN)  
  	LINK_DIRECTORIES(${EO_BIN_DIR}\\lib\\${CMAKE_BUILD_TYPE}
  					 ${ParadisEO-MO_BINARY_DIR}\\lib\\${CMAKE_BUILD_TYPE}
  					)
ENDIF(WIN32 AND NOT CYGWIN) 

######################################################################################



######################################################################################
### 3) Define your targets and link the librairies
######################################################################################

SET (TEST_LIST  t-moBestImprSelect 
       		t-moExponentialCoolingSchedule 
		t-moFirstImprSelect 
   		t-moFitComparator 
    		t-moFitSolContinue 
    		t-moGenSolContinue 
    		t-moHC 
    		t-moHCMoveLoopExpl 
    		t-moILS 
    		t-moImprBestFitAspirCrit 
    		t-moItRandNextMove 
        	t-moLinearCoolingSchedule 
    		t-moLSCheckPoint 
    		t-moNoAspirCrit 
    		t-moNoFitImprSolContinue
    		t-moRandImprSelect 
    		t-moSA  
    		t-moSimpleMoveTabuList 
    		t-moSimpleSolutionTabuList 
    	        t-moSteadyFitSolContinue 
    		t-moTS 
    		t-moTSMoveLoopExpl )
				
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
	 TARGET_LINK_LIBRARIES(${test} ga es eoutils eo)
	ENDFOREACH (test)
	
ENDIF(ENABLE_CMAKE_TESTING)

######################################################################################

