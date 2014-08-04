###############################################################################
##
## CMakeLists file for eo/test/mpi
##
###############################################################################

######################################################################################
### 1) Include the sources
######################################################################################

#include_directories(${EOMPI_SRC_DIR}/src)
#include_directories(${EOSERIAL_SRC_DIR}/src)
#include_directories(${EO_SRC_DIR}/src)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

link_directories(${EO_BIN_DIR}/lib)
link_directories(${EOMPI_BIN_DIR}/lib)
link_directories(${EOSERIAL_BIN_DIR}/lib)

######################################################################################
### 3) Define your targets and link the librairies
######################################################################################

set (TEST_LIST
    t-mpi-parallelApply 
    t-mpi-wrapper
    t-mpi-multipleRoles
    #t-mpi-eval
    #t-mpi-multistart
    t-mpi-distrib-exp
    )

foreach (test ${TEST_LIST})
  set ("T_${test}_SOURCES" "${test}.cpp")
endforeach (test)

set(CMAKE_CXX_COMPILER "mpicxx")
add_definitions(-DWITH_MPI)

if(ENABLE_CMAKE_TESTING)
    foreach (test ${TEST_LIST})
    add_executable(${test} ${T_${test}_SOURCES})
    add_test(${test} ${test})
    target_link_libraries(${test} eoutils eompi eoserial eo)
    install(TARGETS ${test} RUNTIME DESTINATION share/eo/test COMPONENT test)
    endforeach (test)
endif()

######################################################################################
