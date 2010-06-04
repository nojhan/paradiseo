###############################################################################
##
## CMakeLists file for ParadisEO-MOEO/test
##
###############################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${MO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${OLDMO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${ParadisEO-MOEO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${ParadisEO-MOEO_BINARY_DIR}/lib)

######################################################################################



######################################################################################
### 3) Define your targets and link the librairies
######################################################################################

SET (TEST_LIST
		t-moeo
		t-moeoBitVector
		t-moeoRealVector
		t-moeoUnboundedArchive
		t-moeoParetoObjectiveVectorComparator
		t-moeoStrictObjectiveVectorComparator
		t-moeoWeakObjectiveVectorComparator
		t-moeoEpsilonObjectiveVectorComparator
		t-moeoAggregativeComparator
		t-moeoDiversityThenFitnessComparator
		t-moeoFitnessThenDiversityComparator
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
		t-moeoHyperVolumeMetric
		t-moeoHyperVolumeDifferenceMetric
		t-moeoIntVector
		t-moeoImprOnlyBoundedArchive
		t-moeoFitDivBoundedArchive
		t-moeoDetArchiveSelect
		t-moeoASEEA
		t-moeoEpsilonHyperboxArchive
		#t-moeoQuadTreeIndex
		t-moeoQuickUnboundedArchiveIndex
		t-moeoAggregationFitnessAssignment
		t-moeoConstraintFitnessAssignment
		t-moeoChebyshevMetric
		t-moeoChebyshevOrientedMetric
		t-moeoASFAMetric
		t-moeoASFAOrMetric
		t-moeoHC
		t-moeoILS_HC
		t-moeoILS_SA
		t-moeoILS_TS
		t-moeoTS
		t-moeoVNS
		t-moeoVFAS
		t-moeoAnytimeWeightStrategy
		t-moeoQexploreWeightStrategy
		t-moeoAugmentedQexploreWeightStrategy
		t-moeoFixedTimeOneDirectionWeightStrategy
		t-moeoFixedTimeBothDirectionWeightStrategy
		t-moeoDichoWeightStrategy
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
)

FOREACH (test ${TEST_LIST})
	  SET ("T_${test}_SOURCES" "${test}.cpp")
ENDFOREACH (test)



IF(ENABLE_MINIMAL_CMAKE_TESTING)
    SET (MIN_TEST_LIST t-moeoEasyEA)				
    FOREACH (mintest ${MIN_TEST_LIST})
    	  SET ("T_${mintest}_SOURCES" "${mintest}.cpp")
    	  ADD_EXECUTABLE(${mintest} ${T_${mintest}_SOURCES}) 
		  ADD_TEST(${mintest} ${mintest}) 
          TARGET_LINK_LIBRARIES(${mintest} moeo ga es eoutils eo)
    ENDFOREACH (mintest)
   
ELSEIF(ENABLE_CMAKE_TESTING)
	
	FOREACH (test ${TEST_LIST})
		  ADD_EXECUTABLE(${test} ${T_${test}_SOURCES}) 
		  ADD_TEST(${test} ${test}) 
	ENDFOREACH (test)
	
	# Link the librairies
	FOREACH (test ${TEST_LIST})
	 TARGET_LINK_LIBRARIES(${test} moeo ga es eoutils eo)
	ENDFOREACH (test)
	
ENDIF(ENABLE_MINIMAL_CMAKE_TESTING)

######################################################################################

