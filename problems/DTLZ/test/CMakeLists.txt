###############################################################################
##
## CMakeLists file for DTLZ/test
##
###############################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${PARADISEO_EO_SRC_DIR}/src
                    ${PARADISEO_MO_SRC_DIR}/src
                    ${PARADISEO_MOEO_SRC_DIR}/src
                    ${DTLZ_SOURCE_DIR}/src)

######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${PARADISEO_EO_BIN_DIR}/lib)
LINK_DIRECTORIES(${PARADISEO_MOEO_BIN_DIR}/lib)
LINK_DIRECTORIES(${DTLZ_BINARY_DIR}/lib)
######################################################################################



######################################################################################
### 3) Define your targets and link the librairies
######################################################################################

SET (TEST_LIST
	t-DTLZ
	t-DTLZ1Eval
	t-DTLZ2Eval
	t-DTLZ3Eval
	t-DTLZ4Eval
	t-DTLZ5Eval
	t-DTLZ6Eval
	t-DTLZ7Eval
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
	 TARGET_LINK_LIBRARIES(${test} moeo eoutils eo lDTLZ)
	ENDFOREACH (test)


######################################################################################

