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
				  t-EAAsyncIsland
				  t-EASyncIsland
				  t-PSOGlobalBest
				  t-PSOWorstPosition
				  t-Mpdallexit)
				  
SET (TEST_STOP    t-Mpdallexit)
	
FOREACH (test ${TEST_LIST} ${TEST_RUN} ${TEST_STOP})
	  SET ("T_${test}_SOURCES" "${test}.cpp")
ENDFOREACH (test)


IF(ENABLE_CMAKE_TESTING)
	
	# Add the tests
	FOREACH (test ${TEST_LIST} ${TEST_RUN} ${TEST_STOP})
		  ADD_EXECUTABLE(${test} ${T_${test}_SOURCES})  
	ENDFOREACH (test)
	
	FOREACH (test ${TEST_RUN})
		  ADD_TEST(${test} ${test})
	ENDFOREACH (test)
	
	FOREACH (test ${TEST_LIST})
		  ADD_TEST(${test} "mpiexec" "-n" "4" "${CMAKE_CURRENT_BINARY_DIR}/${test}" "@param")
	ENDFOREACH (test)
	
	FOREACH (test ${TEST_STOP})
		  ADD_TEST(${test} ${test})
	ENDFOREACH (test)
	
	# Link the librairies
	FOREACH (test ${TEST_LIST})
	 TARGET_LINK_LIBRARIES(${test} peo ${XML2_LIBS} rmc_mpi ga es eoutils eo)
	ENDFOREACH (test)
	
ENDIF(ENABLE_CMAKE_TESTING)

######################################################################################


######################################################################################
### 5) Windows advanced config - especially for Microsoft Visual Studio 8
######################################################################################

  IF(CMAKE_CXX_COMPILER MATCHES cl)
   IF(NOT WITH_SHARED_LIBS)
     IF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005")
       SET(CMAKE_CXX_FLAGS "/nologo /W3 /Gy")
       SET(CMAKE_CXX_FLAGS_DEBUG "/MTd /Z7 /Od")
       SET(CMAKE_CXX_FLAGS_RELEASE "/MT /O2")
       SET(CMAKE_CXX_FLAGS_MINSIZEREL "/MT /O2")
       SET(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MTd /Z7 /Od")
       SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /SUBSYSTEM:CONSOLE")
       
     ENDIF(CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005")
   ENDIF(NOT WITH_SHARED_LIBS)
  ENDIF(CMAKE_CXX_COMPILER MATCHES cl)
######################################################################################



