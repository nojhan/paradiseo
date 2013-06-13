######################################################################################
### 0) Include headers
######################################################################################

include_directories(${EO_SRC_DIR}/src)
include_directories(${MO_SRC_DIR}/src)
include_directories(${EDO_SRC_DIR}/src)
include_directories(${PROBLEMS_SRC_DIR})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 1) Define test list
######################################################################################

set (TEST_LIST  
        t-moAdaptiveWalkSampling
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
		t-moTS
		t-moILS
		t-moDummyLS
		t-moRandomSearch
		t-moMetropolisHasting
		t-moNeutralHC
		t-moRandomWalk
		t-moRandomNeutralWalk
		t-moIterContinuator
		t-moFitContinuator
		t-moCombinedContinuator
		t-moFullEvalContinuator
		t-moNeighborEvalContinuator
		t-moTimeContinuator
		t-moDummyExplorer
		t-moLocalSearchInit
		t-moSolInit
		t-moEvalCounter
		t-moNeighborBestStat
		t-moCounterStat
		t-moMinusOneCounterStat
		t-moVectorMonitor
		t-moRandomSearchExplorer
		t-moSampling
		t-moDensityOfStatesSampling
		t-moAutocorrelationSampling
		t-moHillClimberSampling
		t-moFDCsampling
		t-moNeutralDegreeSampling
		t-moFitnessCloudSampling
		t-moNeutralWalkSampling
		t-moStatistics
		t-moIndexedVectorTabuList
#		t-moRndIndexedVectorTabuList
		t-moStdFitnessNeighborStat
		t-moStdDevEstimator
		t-moDynSpanCoolingSchedule
		t-moTriki
		t-moTrikiReal
		)

######################################################################################
### 3) Create each test
######################################################################################

foreach (test ${TEST_LIST})
	set ("T_${test}_SOURCES" "${test}.cpp")
    add_executable(${test} ${T_${test}_SOURCES}) 
    add_test(${test} ${test}) 
    target_link_libraries(${test} ga es eoutils eo)
    install(TARGETS ${test} RUNTIME DESTINATION share${INSTALL_SUB_DIR}/mo/test COMPONENT tests)
endforeach (test)


