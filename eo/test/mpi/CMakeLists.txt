###############################################################################
##
## CMakeLists file for eo/test/mpi
##
###############################################################################

######################################################################################
### 1) Include the sources
######################################################################################

MESSAGE("EO SOURCE DIR: ${EO_SOURCE_DIR}")
MESSAGE("OMPI: ${MPI_DIR}")

INCLUDE_DIRECTORIES(${MPI_DIR}/include)
INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

LINK_DIRECTORIES(${EO_BINARY_DIR}/lib)
LINK_DIRECTORIES(${MPI_DIR}/lib)

######################################################################################
### 3) Define your targets and link the librairies
######################################################################################

SET (TEST_LIST
    t-mpi-parallelApply
    t-mpi-wrapper
    t-mpi-multipleRoles
    t-mpi-eval
    t-mpi-multistart
    t-mpi-distrib-exp
    )

FOREACH (test ${TEST_LIST})
  SET ("T_${test}_SOURCES" "${test}.cpp")
ENDFOREACH (test)

SET(CMAKE_CXX_COMPILER "mpicxx")
ADD_DEFINITIONS(-DWITH_MPI)

IF(ENABLE_CMAKE_TESTING)
    FOREACH (test ${TEST_LIST})
    ADD_EXECUTABLE(${test} ${T_${test}_SOURCES})
    ADD_TEST(${test} ${test})
    TARGET_LINK_LIBRARIES(${test} eoutils eompi eoserial eo)
    INSTALL(TARGETS ${test} RUNTIME DESTINATION share/eo/test COMPONENT test)
    ENDFOREACH (test)
ENDIF()

######################################################################################
