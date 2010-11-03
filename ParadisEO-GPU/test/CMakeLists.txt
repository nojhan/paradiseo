###############################################################################
##
## CMakeLists file for OneMax Example/test
##
###############################################################################


######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(
  # include CUDA source directory
  ${CUDA_SRC_DIR}
  # include NVIDIA source directory
  ${NVIDIA_SRC_DIR}
  # include EO source directory
  ${PARADISEO_EO_SRC_DIR}/src
  # include MO source directory
  ${PARADISEO_MO_SRC_DIR}/src
  # include problems  directory
  ${PARADISEO_PROBLEMS_SRC_DIR}
  # include GPU  directory
  ${PARADISEO_GPU_SRC_DIR}
  # include your source directory
  ${CMAKE_CURRENT_SOURCE_DIR}/../src
  )

######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${PARADISEO_EO_BIN_DIR}/lib  ${NVIDIA_LIB_DIR} ${CUDA_LIB_DIR} )	


######################################################################################

######################################################################################
### 3) Define your targets and link the librairies
######################################################################################

SET (TEST_LIST
	       t-memory
           t-moCudaBitVector
           t-moCudaIntVector
	       t-moCudaBitNeighbor
           t-OneMaxIncrEval
           t-EvalOneMax
         # t-moCudaKswapNeighborhood
		)
				
FOREACH (test ${TEST_LIST})
	  SET ("T_${test}_SOURCES" "${test}.cu")
ENDFOREACH (test)


IF(ENABLE_CMAKE_TESTING)
	
	# Add the tests
	FOREACH (test ${TEST_LIST})
		  CUDA_ADD_EXECUTABLE(${test} ${T_${test}_SOURCES}) 
		  ADD_TEST(${test} ${test}) 
	ENDFOREACH (test)
	
	# Link the librairies
	FOREACH (test ${TEST_LIST})
	 	TARGET_LINK_LIBRARIES(${test} ga es eoutils eo)
	ENDFOREACH (test)
	
ENDIF(ENABLE_CMAKE_TESTING)

######################################################################################

