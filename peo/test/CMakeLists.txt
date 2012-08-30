######################################################################################
### 0) Include headers
######################################################################################

include_directories(${EO_SRC_DIR}/src)
include_directories(${MO_SRC_DIR}/src)
include_directories(${MOEO_SRC_DIR}/src)
include_directories(${PEO_SRC_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 1) Set the compiler for MPI
######################################################################################

set(CMAKE_CXX_COMPILER mpicxx)

######################################################################################
### 2) Define test list
######################################################################################

set(TEST_RUN t-Mpdboot)
set(TEST_STOP t-Mpdallexit)
set(TEST_LIST
        t-ParallelEval
        t-ParallelTransform
	    t-MultiStart
	    t-PSOSelect
	    t-EAAsyncIsland
	    t-EASyncIsland
	    t-PSOGlobalBest
	    t-PSOWorstPosition
	    )

######################################################################################
### 3) Define your targets and link the librairies
######################################################################################
execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${CMAKE_CURRENT_SOURCE_DIR}/param
			${CMAKE_CURRENT_BINARY_DIR}/param
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${CMAKE_CURRENT_SOURCE_DIR}/schema.xml
			${CMAKE_CURRENT_BINARY_DIR}/schema.xml)
			        
foreach (test ${TEST_RUN} ${TEST_LIST} ${TEST_STOP})
    set("T_${test}_SOURCES" "${test}.cpp")
    add_executable(${test} ${T_${test}_SOURCES})
    install(TARGETS ${test} RUNTIME DESTINATION local/share${INSTALL_SUB_DIR}/peo/test COMPONENT tests)
endforeach (test)
	
foreach (testrun ${TEST_RUN})
    add_test(${testrun} ${testrun})
endforeach (testrun)
	
foreach (test ${TEST_LIST})
    add_test(${test} "mpiexec" "-n" "4" "${CMAKE_CURRENT_BINARY_DIR}/${test}" "@param")
    target_link_libraries(${test} peo xml2 rmc_mpi ga es eoutils eo peo)
    
endforeach (test)
	
foreach (teststop ${TEST_STOP})
    add_test(${teststop} ${teststop})
endforeach (teststop)
		
