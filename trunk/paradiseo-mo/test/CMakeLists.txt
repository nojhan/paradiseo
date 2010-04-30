###############################################################################
##
## CMakeLists file for ParadisEO-MO/test
##
###############################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${MO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${PROBLEMS_SRC_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${EO_BIN_DIR}/lib)

######################################################################################

######################################################################################
### 3) Define your targets and link the librairies
######################################################################################



SET (TEST_LIST
		t-moNeighbor
		t-moBitNeighbor
		t-moOrderNeighborhood
		t-moFullEvalByCopy
		t-moFullEvalByModif
		t-moNeighborComparator
		t-moSolNeighborComparator
		t-moTrueContinuator
		t-moRndWithoutReplNeighborhood
		t-moRndWithReplNeighborhood
		t-moFitnessStat
		t-moDistanceStat
		t-moNeighborhoodStat
		t-moCounterMonitorSaver
		t-moSolutionStat
		t-moCheckpoint
		t-moDummyMemory
		t-moSolVectorTabuList
		t-moBestImprAspiration
		t-moSimpleHCexplorer
		t-moRandomBestHCexplorer
		t-moNeutralHCexplorer
		t-moFirstImprHCexplorer
		t-moRandomWalkExplorer
		t-moMetropolisHastingExplorer
		t-moRandomNeutralWalkExplorer
		t-moTSexplorer
		t-moForwardVariableNeighborhood
		t-moSolComparator
		t-moDummyEval
		t-moDummyNeighbor
		t-moDummyNeighborhood
		t-moSimpleCoolingSchedule
		t-moAlwaysAcceptCrit
		t-moBetterAcceptCrit
		t-moCountMoveMemory
		t-moMonOpPerturb
		t-moRestartPerturb
		t-moNeighborhoodPerturb
		t-moSAexplorer
		t-moSA
		t-moLocalSearch
		t-moILSexplorer
		t-moSimpleHC
		t-moFirstImprHC
		t-moRandomBestHC
		t-moNeighborVectorTabuList
		t-moMonOpDiversification
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
	 	TARGET_LINK_LIBRARIES(${test} ga es eoutils eo)
	ENDFOREACH (test)
	
ENDIF(ENABLE_CMAKE_TESTING)

######################################################################################

