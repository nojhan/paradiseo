###############################################################################
##
## CMakeLists file for ParadisEO-MO/test
##
###############################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${PARADISEO_EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${NEWMO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${PARADISEO_EO_BIN_DIR}/lib)

######################################################################################

######################################################################################
### 3) Define your targets and link the librairies
######################################################################################

SET (TEST_LIST
		t-moNeighbor
		t-moBitNeighbor
		t-moBitNeighborhood
		t-moFullEvalByCopy
		t-moFullEvalByModif
		t-moSimpleHCexplorer)
				
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

