###############################################################################
##
## CMakeLists file for ParadisEO-MOEO/test
##
###############################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${PARADISEO_EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${PARADISEO_MO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${PARADISEO_MOEO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})
INCLUDE_DIRECTORIES(${SOURCES_DIR})


######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${PARADISEO_EO_BIN_DIR}/lib ${PARADISEO_MOEO_BIN_DIR}/lib)

######################################################################################



######################################################################################
### 3) Define your targets and link the librairies
######################################################################################

SET (TEST_LIST
		t-moeoUnifiedDominanceBasedLS		
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
 TARGET_LINK_LIBRARIES(${test} eoutils eo moeo)
ENDFOREACH (test)

######################################################################################

