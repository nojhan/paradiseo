######################################################################################
### 0) Include headers
######################################################################################

include_directories(${EO_SRC_DIR}/src)
include_directories(${MO_SRC_DIR}/src)
include_directories(${MOEO_SRC_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})


######################################################################################
### 1) Define test list
######################################################################################

set(TEST_LIST
		t-moeo
		t-moeoBitVector
		t-moeoRealVector
		t-moeoUnboundedArchive
		t-moeoParetoObjectiveVectorComparator
		t-moeoStrictObjectiveVectorComparator
		t-moeoWeakObjectiveVectorComparator
		t-moeoEpsilonObjectiveVectorComparator
		#t-moeoAggregativeComparator # TEMPORARLY BYPASSED
		#t-moeoDiversityThenFitnessComparator # IDEM.
		#t-moeoFitnessThenDiversityComparator # IDEM
		t-moeoAchievementFitnessAssignment
		t-moeoExpBinaryIndicatorBasedFitnessAssignment
		t-moeoCrowdingDiversityAssignment
		t-moeoSharingDiversityAssignment
		t-moeoIBEA
		t-moeoNSGA
		t-moeoNSGAII
		t-moeoSEEA
		t-moeoMax3Obj
		t-moeoEasyEA
		t-moeoDominanceCountFitnessAssignment
		t-moeoDominanceRankFitnessAssignment
		t-moeoDominanceCountRankingFitnessAssignment
		t-moeoDominanceDepthFitnessAssignment
		t-moeoNearestNeighborDiversityAssignment
		t-moeoSPEA2Archive
		t-moeoSPEA2
		t-moeoDominanceMatrix
		t-moeoVecVsVecAdditiveEpsilonBinaryMetric
		t-moeoVecVsVecMultiplicativeEpsilonBinaryMetric
		t-moeoHypervolumeBinaryMetric
		t-moeoHyperVolumeMetric
		t-moeoHyperVolumeDifferenceMetric
		t-moeoIntVector
		t-moeoImprOnlyBoundedArchive
		t-moeoFitDivBoundedArchive
		t-moeoDetArchiveSelect
		t-moeoASEEA
		t-moeoEpsilonHyperboxArchive
		#t-moeoQuadTreeIndex
		#t-moeoQuickUnboundedArchiveIndex
		t-moeoAggregationFitnessAssignment
		t-moeoConstraintFitnessAssignment
		t-moeoChebyshevMetric
		t-moeoChebyshevOrientedMetric
		t-moeoASFAMetric
		t-moeoASFAOrMetric
		t-moeoExhaustiveNeighborhoodExplorer
		t-moeoFirstImprovingNeighborhoodExplorer
		t-moeoSimpleSubNeighborhoodExplorer
		t-moeoNoDesimprovingNeighborhoodExplorer
		t-moeoPLS1
		t-moeoPLS2
		t-moeoExhaustiveUnvisitedSelect
		t-moeoNumberUnvisitedSelect
		t-moeoDMLSMonOp
		t-moeoDMLSGenUpdater
		t-moeo2DMinHypervolumeArchive
)

######################################################################################
### 3) Create each test
######################################################################################

foreach (test ${TEST_LIST})
    set("T_${test}_SOURCES" "${test}.cpp")
    add_executable(${test} ${T_${test}_SOURCES}) 
    add_test(${test} ${test}) 
    target_link_libraries(${test} moeo ga es eoutils eo)
    install(TARGETS ${test} RUNTIME DESTINATION share${INSTALL_SUB_DIR}/moeo/test COMPONENT tests)
endforeach (test)
