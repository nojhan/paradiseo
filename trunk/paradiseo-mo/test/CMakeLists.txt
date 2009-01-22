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

LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${ParadisEO-MO_BINARY_DIR}/lib)

######################################################################################



######################################################################################
### 3) Define your targets and link the librairies
######################################################################################

SET (TEST_LIST  t-moBestImprSelect 
               	t-moFirstImprSelect 
           	t-moFitComparator 
    		t-moFitSolContinue 
    		t-moGenSolContinue 
    		t-moGeometricCoolingSchedule 
		t-moHC 
		t-moHC_2
    		t-moHCMoveLoopExpl 
    		t-moILS
		t-moILS_HC
		t-moILS_TS
		t-moILS_SA
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
		t-moTS_2
    	    	t-moTSMoveLoopExpl
    	    	t-moTA
    	    	t-moVNS
    	    	)
				
FOREACH (test ${TEST_LIST})
	  SET ("T_${test}_SOURCES" "${test}.cpp")
ENDFOREACH (test)


IF(ENABLE_MINIMAL_CMAKE_TESTING)
    SET (MIN_TEST_LIST t-moHC)				
    FOREACH (mintest ${MIN_TEST_LIST})
    	  SET ("T_${mintest}_SOURCES" "${mintest}.cpp")
    	  ADD_EXECUTABLE(${mintest} ${T_${mintest}_SOURCES}) 
		  ADD_TEST(${mintest} ${mintest}) 
    ENDFOREACH (mintest)
   
ELSEIF(ENABLE_CMAKE_TESTING)
	
	# Add the tests
	FOREACH (test ${TEST_LIST})
		  ADD_EXECUTABLE(${test} ${T_${test}_SOURCES}) 
		  ADD_TEST(${test} ${test}) 
	ENDFOREACH (test)
	
	# Link the librairies
	FOREACH (test ${TEST_LIST})
	 TARGET_LINK_LIBRARIES(${test} ga es eoutils eo)
	ENDFOREACH (test)
	
ENDIF(ENABLE_MINIMAL_CMAKE_TESTING)

######################################################################################

