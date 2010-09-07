IF(COMMAND cmake_policy)
      CMAKE_POLICY(SET CMP0003 NEW)
ENDIF(COMMAND cmake_policy)
######################################################################################
### 1) Definitions (required for tsp target)
######################################################################################

SET(TSP_SRC_DIR ${OLDMO_SRC_DIR}/tutorial/examples/tsp CACHE PATH "TSP src directory")
SET(TSP_BIN_DIR ${OLDMO_BIN_DIR}/tutorial/examples/tsp CACHE PATH "TSP binary directory")

SET(FUNCTION_SRC_DIR ${OLDMO_SRC_DIR}/tutorial/examples/function CACHE PATH "FUNCTION src directory")
SET(FUNCTION_BIN_DIR ${OLDMO_BIN_DIR}/tutorial/examples/function CACHE PATH "FUNCTION binary directory")

######################################################################################

######################################################################################
### 2) How to manage copy of benchs ?
######################################################################################

IF(ENABLE_CMAKE_EXAMPLE OR CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")
	SET(BENCH_LIST
		berlin52.tsp
		eil101.tsp
		pr2392.tsp
		rl5915.tsp
		usa13509.tsp
	)
	
	FOREACH (bench ${BENCH_LIST})
			EXECUTE_PROCESS(
				COMMAND ${CMAKE_COMMAND} -E copy_if_different
	      			${OLDMO_SRC_DIR}/tutorial/examples/tsp/benchs/${bench}
	    			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/${bench}
	    	)
	ENDFOREACH (bench)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${OLDMO_SRC_DIR}/tutorial/Lesson1/param
			${OLDMO_BIN_DIR}/tutorial/Lesson1/param
    	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${OLDMO_SRC_DIR}/tutorial/Lesson2/param
			${OLDMO_BIN_DIR}/tutorial/Lesson2/param
    	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${OLDMO_SRC_DIR}/tutorial/Lesson3/param
			${OLDMO_BIN_DIR}/tutorial/Lesson3/param
    	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${OLDMO_SRC_DIR}/tutorial/Lesson4/param
			${OLDMO_BIN_DIR}/tutorial/Lesson4/param
    	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${OLDMO_SRC_DIR}/tutorial/Lesson5/param
			${OLDMO_BIN_DIR}/tutorial/Lesson5/param
    	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${OLDMO_SRC_DIR}/tutorial/HybridLesson/param
			${OLDMO_BIN_DIR}/tutorial/HybridLesson/param
	)
ENDIF(ENABLE_CMAKE_EXAMPLE OR CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")

######################################################################################
### 3) Where must cmake go now ?
######################################################################################

MACRO(ADD_COMMANDS_OLDMO)
	ADD_CUSTOM_COMMAND(
	OUTPUT 
    	${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
    	${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
    	${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
    	${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
    	${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${OLDMO_SRC_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
		${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${OLDMO_SRC_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
    	${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${OLDMO_SRC_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
    	${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${OLDMO_SRC_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
    	${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${OLDMO_SRC_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
    	${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
	)
	ADD_CUSTOM_COMMAND(
		OUTPUT ${OLDMO_BIN_DIR}/tutorial/Lesson1/param
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${OLDMO_SRC_DIR}/tutorial/Lesson1/param
			${OLDMO_BIN_DIR}/tutorial/Lesson1
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${OLDMO_BIN_DIR}/tutorial/Lesson2/param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
	    	${OLDMO_SRC_DIR}/tutorial/Lesson2/param
	    	${OLDMO_BIN_DIR}/tutorial/Lesson2
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${OLDMO_BIN_DIR}/tutorial/Lesson3/param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
	    	${OLDMO_SRC_DIR}/tutorial/Lesson3/param
	    	${OLDMO_BIN_DIR}/tutorial/Lesson3
	)
	ADD_CUSTOM_COMMAND(
    OUTPUT ${OLDMO_BIN_DIR}/tutorial/Lesson4/param
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${OLDMO_SRC_DIR}/tutorial/Lesson4/param
    	${OLDMO_BIN_DIR}/tutorial/Lesson4
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${OLDMO_BIN_DIR}/tutorial/Lesson5/param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
	    	${OLDMO_SRC_DIR}/tutorial/Lesson5/param
	    	${OLDMO_BIN_DIR}/tutorial/Lesson5
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${OLDMO_BIN_DIR}/tutorial/Lesson6/param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
	    	${OLDMO_SRC_DIR}/tutorial/HybridLesson/param
	    	${OLDMO_BIN_DIR}/tutorial/HybridLesson
	)
ENDMACRO(ADD_COMMANDS_OLDMO)

MACRO(ADD_TARGET_OLDMO n)
	IF(${n} STREQUAL "lesson1")
		ADD_CUSTOM_TARGET(lesson1 DEPENDS 
			${OLDMO_BIN_DIR}/tutorial/Lesson1/param
			${OLDMO_BIN_DIR}/tutorial/Lesson1/hill_climbing
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
		)
	ELSEIF(${n} STREQUAL "lesson2")	
		ADD_CUSTOM_TARGET(lesson2 DEPENDS 
			${OLDMO_BIN_DIR}/tutorial/Lesson2/param
			${OLDMO_BIN_DIR}/tutorial/Lesson2/tabu_search
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
		)
	ELSEIF(${n} STREQUAL "lesson3")			
		ADD_CUSTOM_TARGET(lesson3 DEPENDS
			${OLDOLDMO_BIN_DIR}/tutorial/Lesson3/param
			${OLDMO_BIN_DIR}/tutorial/Lesson3/simulated_annealing
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
		)
	ELSEIF(${n} STREQUAL "lesson4")				
		ADD_CUSTOM_TARGET(lesson4 DEPENDS 
			${OLDMO_BIN_DIR}/tutorial/Lesson4/param
			${OLDMO_BIN_DIR}/tutorial/Lesson4/iterated_local_search
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
		)
	ELSEIF(${n} STREQUAL "lesson5")				
		ADD_CUSTOM_TARGET(lesson5 DEPENDS 
			${OLDMO_BIN_DIR}/tutorial/Lesson5/param
			${OLDMO_BIN_DIR}/tutorial/Lesson5/variable_neighborhood_search
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
		)
	ELSEIF(${n} STREQUAL "hybridlesson")		
		ADD_CUSTOM_TARGET(hybridlesson DEPENDS 
			${OLDMO_BIN_DIR}/tutorial/HybridLesson/param
			${OLDMO_BIN_DIR}/tutorial/HybridLesson/hybrid_ga
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
		)
	ELSEIF(${n} STREQUAL "install")		
		ADD_CUSTOM_TARGET(install DEPENDS 
			${OLDMO_BIN_DIR}/tutorial/Lesson1/param
			${OLDMO_BIN_DIR}/tutorial/Lesson2/param
			${OLDMO_BIN_DIR}/tutorial/Lesson3/param
			${OLDMO_BIN_DIR}/tutorial/Lesson4/param
			${OLDMO_BIN_DIR}/tutorial/Lesson5/param
			${OLDMO_BIN_DIR}/tutorial/HybridLesson/param
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
		)
	ELSEIF(${n} STREQUAL "benchs")
		ADD_CUSTOM_TARGET(benchs DEPENDS
			${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
	    	${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
	    	${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
	    	${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
	    	${OLDMO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
		)	
	ENDIF(${n} STREQUAL "lesson1")
ENDMACRO(ADD_TARGET_OLDMO)

IF(NOT CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" AND  NOT CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")
	ADD_COMMANDS_OLDMO()
	ADD_TARGET_OLDMO(install)
	ADD_TARGET_OLDMO(benchs)
ENDIF(NOT CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" AND NOT CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008" OR CMAKE_GENERATOR STREQUAL "Visual Studio 10")

ADD_SUBDIRECTORY(examples)
ADD_SUBDIRECTORY(Lesson1)
ADD_SUBDIRECTORY(Lesson2)
ADD_SUBDIRECTORY(Lesson3)
ADD_SUBDIRECTORY(Lesson4)
ADD_SUBDIRECTORY(Lesson5)
ADD_SUBDIRECTORY(HybridLesson)
