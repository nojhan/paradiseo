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
		t-moeoArchive
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
		t-moeoEasyEA
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
	 TARGET_LINK_LIBRARIES(${test} moeo ga es eoutils eo)
	ENDFOREACH (test)
	
ENDIF(ENABLE_CMAKE_TESTING)

######################################################################################

