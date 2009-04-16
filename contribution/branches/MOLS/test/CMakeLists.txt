SET (TEST_LIST
		t-moeoExhaustiveUnvisitedSelect
		t-moeoNumberUnvisitedSelect
		t-moeoExhaustiveNeighborhoodExplorer
		t-moeoSimpleSubNeighborhoodExplorer
		t-moeoFirstImprovingNeighborhoodExplorer
		t-moeoRecursiveFirstImprovingNeighborhoodExplorer
		t-moeoUnifiedDominanceBasedLS
		t-moeoDMLSGenUpdater
		t-moeoDMLSMonOp
	)

FOREACH (test ${TEST_LIST})
	  SET ("T_${test}_SOURCES" "${test}.cpp")
ENDFOREACH (test)

	
	FOREACH (test ${TEST_LIST})
		  ADD_EXECUTABLE(${test} ${T_${test}_SOURCES}) 
		  ADD_TEST(${test} ${test}) 
	ENDFOREACH (test)
	
	# Link the librairies
	FOREACH (test ${TEST_LIST})
	 TARGET_LINK_LIBRARIES(${test} moeo ga es eoutils eo)
	ENDFOREACH (test)

######################################################################################

