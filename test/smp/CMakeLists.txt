######################################################################################
### 0) Include headers
######################################################################################

#include_directories(${EO_SRC_DIR}/src)
#include_directories(${SMP_SRC_DIR}/src)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 1) Define test list
######################################################################################

set (TEST_LIST  
        t-smpScheduler
        t-smpMW_eoEasyEA
        t-smpMW_eoEasyPSO
        t-smpMW_eoSyncEasyPSO
        t-smpIsland
        t-smpTopo
        t-smpMI_Homogeneous
        t-smpMI_Heterogeneous
        t-smpMI_Wrapper
        t-smpCustomTopo
		)

######################################################################################
### 3) Create each test
######################################################################################

foreach (test ${TEST_LIST})
	set ("T_${test}_SOURCES" "${test}.cpp")
    add_executable(${test} ${T_${test}_SOURCES})
    target_link_libraries(${test} smp eo eoutils)
    add_test(${test} ${test}) 
    install(TARGETS ${test} RUNTIME DESTINATION share/${PROJECT_TAG}/smp/test COMPONENT tests)
endforeach (test)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/t-data.dat
    ${CMAKE_CURRENT_BINARY_DIR}/t-data.dat)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/t-data.dat DESTINATION share/${PROJECT_TAG}/smp/test COMPONENT tests)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data-topo-bool
    ${CMAKE_CURRENT_BINARY_DIR}/data-topo-bool)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data-topo-bool DESTINATION share/${PROJECT_TAG}/smp/test COMPONENT tests)

execute_process(
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    ${CMAKE_CURRENT_SOURCE_DIR}/data-topo-stoch
    ${CMAKE_CURRENT_BINARY_DIR}/data-topo-stoch)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/data-topo-stoch DESTINATION share/${PROJECT_TAG}/smp/test COMPONENT tests)
