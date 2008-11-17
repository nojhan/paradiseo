###############################################################################
##
## CMakeLists file for ParadisEO-MOEO/test
##
###############################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
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
		t-moeoAggregativeComparator
		t-moeoDiversityThenFitnessComparator
		t-moeoFitnessThenDiversityComparator
		t-moeoAchievementFitnessAssignment
		t-moeoExpBinaryIndicatorBasedFitnessAssignment
		t-moeoFastNonDominatedSortingFitnessAssignment
		t-moeoCrowdingDiversityAssignment
		t-moeoSharingDiversityAssignment
		t-moeoIBEA
		t-moeoNSGA
		t-moeoNSGAII
		t-moeoMax3Obj
		t-moeoEasyEA
		t-moeoDominanceCountFitnessAssignment
		t-moeoDominanceRankFitnessAssignment
		t-moeoDominanceCountRankingFitnessAssignment
		t-moeoNearestNeighborDiversityAssignment
		t-moeoSPEA2Archive
		t-moeoSPEA2
		t-moeoDominanceMatrix
		t-moeoVecVsVecAdditiveEpsilonBinaryMetric
		t-moeoVecVsVecMultiplicativeEpsilonBinaryMetric
		t-moeoHyperVolumeMetric
		t-moeoHyperVolumeDifferenceMetric
		t-moeoIntVector
		t-moeoBoundedArchive
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

