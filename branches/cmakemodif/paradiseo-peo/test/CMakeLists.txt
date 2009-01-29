###############################################################################
##
## CMakeLists file for ParadisEO-PEO/test
##
###############################################################################



######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${MO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${MOEO_SRC_DIR}/src)
INCLUDE_DIRECTORIES(${ParadisEO-PEO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

INCLUDE(CheckLibraryExists)

SET (CMAKE_CXX_COMPILER mpicxx)

ADD_CUSTOM_TARGET(install ALL DEPENDS ${ParadisEO-PEO_SOURCE_DIR}/test/param ${ParadisEO-PEO_SOURCE_DIR}/test/schema.xml)
ADD_CUSTOM_COMMAND(
    TARGET install
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/test/param
      ${ParadisEO-PEO_BINARY_DIR}/test)
ADD_CUSTOM_COMMAND(
    TARGET install
    PRE_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${ParadisEO-PEO_SOURCE_DIR}/test/schema.xml
      ${ParadisEO-PEO_BINARY_DIR}/test)
######################################################################################


######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${EO_BIN_DIR}/lib ${ParadisEO-PEO_BINARY_DIR}/lib )

######################################################################################



######################################################################################
### 3) Define your targets and link the librairies
######################################################################################

SET (TEST_RUN    t-Mpdboot)
					  

SET (TEST_LIST	  t-ParallelEval
				  t-ParallelTransform
				  t-MultiStart
				  t-PSOSelect
				  #t-EAAsyncIsland
				  #t-EASyncIsland
				  t-PSOGlobalBest
				  t-PSOWorstPosition)

				  
SET (TEST_STOP    t-Mpdallexit)
	
FOREACH (test ${TEST_RUN} ${TEST_LIST} ${TEST_STOP})
	  SET ("T_${test}_SOURCES" "${test}.cpp")
ENDFOREACH (test)


IF(ENABLE_MINIMAL_CMAKE_TESTING)


	SET (MIN_TEST_LIST t-MultiStart)
   
	FOREACH (testrun ${TEST_RUN})
          ADD_EXECUTABLE(${testrun} ${T_${testrun}_SOURCES}) 
		  ADD_TEST(${testrun} ${testrun})
	ENDFOREACH (testrun)				
   
    	FOREACH (mintest ${MIN_TEST_LIST})
    	  SET ("T_${mintest}_SOURCES" "${mintest}.cpp")
    	  ADD_EXECUTABLE(${mintest} ${T_${mintest}_SOURCES}) 
		  ADD_TEST(${mintest} "mpiexec" "-n" "4" "${CMAKE_CURRENT_BINARY_DIR}/${mintest}" "@param") 
		  TARGET_LINK_LIBRARIES(${mintest} peo ${XML2_LIBS} rmc_mpi ga es eoutils eo peo)
    	ENDFOREACH (mintest)
    
    	FOREACH (teststop ${TEST_STOP})
          ADD_EXECUTABLE(${teststop} ${T_${teststop}_SOURCES}) 
		  ADD_TEST(${teststop} ${teststop})
	ENDFOREACH (teststop)   
   
ELSEIF(ENABLE_CMAKE_TESTING)
	
	FOREACH (test ${TEST_RUN} ${TEST_LIST} ${TEST_STOP})
		  ADD_EXECUTABLE(${test} ${T_${test}_SOURCES})  
	ENDFOREACH (test)
	
	FOREACH (testrun ${TEST_RUN})
		  ADD_TEST(${testrun} ${testrun})
	ENDFOREACH (testrun)
	
	FOREACH (test ${TEST_LIST})
		  ADD_TEST(${test} "mpiexec" "-n" "4" "${CMAKE_CURRENT_BINARY_DIR}/${test}" "@param")
	ENDFOREACH (test)
	
	FOREACH (teststop ${TEST_STOP})
		  ADD_TEST(${teststop} ${teststop})
	ENDFOREACH (teststop)

	FOREACH (test ${TEST_LIST})
	 TARGET_LINK_LIBRARIES(${test} peo ${XML2_LIBS} rmc_mpi ga es eoutils eo peo)
	ENDFOREACH (test) 
	
ENDIF(ENABLE_MINIMAL_CMAKE_TESTING)

######################################################################################
