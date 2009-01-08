
######################################################################################
### 1) Definitions (required for tsp target)
######################################################################################

SET(TSP_SRC_DIR ${MO_SRC_DIR}/tutorial/examples/tsp CACHE PATH "TSP src directory")
SET(TSP_BIN_DIR ${MO_BIN_DIR}/tutorial/examples/tsp CACHE PATH "TSP binary directory")

SET(FUNCTION_SRC_DIR ${MO_SRC_DIR}/tutorial/examples/function CACHE PATH "FUNCTION src directory")
SET(FUNCTION_BIN_DIR ${MO_BIN_DIR}/tutorial/examples/function CACHE PATH "FUNCTION binary directory")

######################################################################################

######################################################################################
### 2) How to manage copy of benchs ?
######################################################################################

IF(ENABLE_CMAKE_EXAMPLE OR CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008")
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
	      			${MO_SRC_DIR}/tutorial/examples/tsp/benchs/${bench}
	    			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/${bench}
	    	)
	ENDFOREACH (bench)
	EXECUTE_PROCESS(
	   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/param
			${MO_BIN_DIR}/tutorial/Lesson1/param
    	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson2/param
			${MO_BIN_DIR}/tutorial/Lesson2/param
    	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson3/param
			${MO_BIN_DIR}/tutorial/Lesson3/param
    	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson4/param
			${MO_BIN_DIR}/tutorial/Lesson4/param
    	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson5/param
			${MO_BIN_DIR}/tutorial/Lesson5/param
    	COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson6/param
			${MO_BIN_DIR}/tutorial/Lesson6/param
	)
ENDIF(ENABLE_CMAKE_EXAMPLE OR CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" OR CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008")

######################################################################################
### 3) Where must cmake go now ?
######################################################################################

MACRO(ADD_COMMANDS_MO)
	ADD_CUSTOM_COMMAND(
	OUTPUT 
    	${MO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
    	${MO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
    	${MO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
    	${MO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
    	${MO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${MO_SRC_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
		${MO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${MO_SRC_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
    	${MO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${MO_SRC_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
    	${MO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${MO_SRC_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
    	${MO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${MO_SRC_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
    	${MO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
	)
	ADD_CUSTOM_COMMAND(
		OUTPUT ${MO_BIN_DIR}/tutorial/Lesson1/param
		COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${MO_SRC_DIR}/tutorial/Lesson1/param
			${MO_BIN_DIR}/tutorial/Lesson1
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson2/param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
	    	${MO_SRC_DIR}/tutorial/Lesson2/param
	    	${MO_BIN_DIR}/tutorial/Lesson2
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson3/param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
	    	${MO_SRC_DIR}/tutorial/Lesson3/param
	    	${MO_BIN_DIR}/tutorial/Lesson3
	)
	ADD_CUSTOM_COMMAND(
    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson4/param
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    	${MO_SRC_DIR}/tutorial/Lesson4/param
    	${MO_BIN_DIR}/tutorial/Lesson4
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson5/param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
	    	${MO_SRC_DIR}/tutorial/Lesson5/param
	    	${MO_BIN_DIR}/tutorial/Lesson5
	)
	ADD_CUSTOM_COMMAND(
	    OUTPUT ${MO_BIN_DIR}/tutorial/Lesson6/param
	    COMMAND ${CMAKE_COMMAND} -E copy_if_different
	    	${MO_SRC_DIR}/tutorial/Lesson6/param
	    	${MO_BIN_DIR}/tutorial/Lesson6
	)
ENDMACRO(ADD_COMMANDS_MO)

MACRO(ADD_TARGET_MO n)
	IF(${n} STREQUAL "lesson1")
		ADD_CUSTOM_TARGET(lesson1 DEPENDS
			${MO_BIN_DIR}/tutorial/Lesson1/param
			${MO_BIN_DIR}/tutorial/Lesson1/minimize_function
		)
	ELSEIF(${n} STREQUAL "lesson2")
		ADD_CUSTOM_TARGET(lesson2 DEPENDS 
			${MO_BIN_DIR}/tutorial/Lesson2/param
			${MO_BIN_DIR}/tutorial/Lesson2/hill_climbing
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
		)
	ELSEIF(${n} STREQUAL "lesson3")	
		ADD_CUSTOM_TARGET(lesson3 DEPENDS 
			${MO_BIN_DIR}/tutorial/Lesson3/param
			${MO_BIN_DIR}/tutorial/Lesson3/tabu_search
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
		)
	ELSEIF(${n} STREQUAL "lesson4")			
		ADD_CUSTOM_TARGET(lesson4 DEPENDS
			${MO_BIN_DIR}/tutorial/Lesson4/param
			${MO_BIN_DIR}/tutorial/Lesson4/simulated_annealing
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
		)
	ELSEIF(${n} STREQUAL "lesson5")				
		ADD_CUSTOM_TARGET(lesson5 DEPENDS 
			${MO_BIN_DIR}/tutorial/Lesson5/param
			${MO_BIN_DIR}/tutorial/Lesson5/iterated_local_search
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
		)
	ELSEIF(${n} STREQUAL "lesson6")		
		ADD_CUSTOM_TARGET(lesson6 DEPENDS 
			${MO_BIN_DIR}/tutorial/Lesson6/param
			${MO_BIN_DIR}/tutorial/Lesson6/hybrid_ga
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
		)
	ELSEIF(${n} STREQUAL "install")		
		ADD_CUSTOM_TARGET(install DEPENDS 
			${MO_BIN_DIR}/tutorial/Lesson1/param
			${MO_BIN_DIR}/tutorial/Lesson2/param
			${MO_BIN_DIR}/tutorial/Lesson3/param
			${MO_BIN_DIR}/tutorial/Lesson4/param
			${MO_BIN_DIR}/tutorial/Lesson5/param
			${MO_BIN_DIR}/tutorial/Lesson6/param
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
		)
	ELSEIF(${n} STREQUAL "benchs")
		ADD_CUSTOM_TARGET(benchs DEPENDS
			${MO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
	    	${MO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
	    	${MO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
	    	${MO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
	    	${MO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
		)	
	ENDIF(${n} STREQUAL "lesson1")
ENDMACRO(ADD_TARGET_MO)

IF(NOT CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" AND  NOT CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008")
	ADD_COMMANDS_MO()
	ADD_TARGET_MO(install)
	ADD_TARGET_MO(benchs)
ENDIF(NOT CMAKE_GENERATOR STREQUAL "Visual Studio 8 2005" AND NOT CMAKE_GENERATOR STREQUAL "Visual Studio 9 2008")

ADD_SUBDIRECTORY(examples)
ADD_SUBDIRECTORY(Lesson1)
ADD_SUBDIRECTORY(Lesson2)
ADD_SUBDIRECTORY(Lesson3)
ADD_SUBDIRECTORY(Lesson4)
ADD_SUBDIRECTORY(Lesson5)
ADD_SUBDIRECTORY(Lesson6)
