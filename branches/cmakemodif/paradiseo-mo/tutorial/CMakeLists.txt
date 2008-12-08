
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

IF(ENABLE_CMAKE_COPY)
	EXECUTE_PROCESS(
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
ENDIF(ENABLE_CMAKE_COPY)

######################################################################################
### 3) Where must cmake go now ?
######################################################################################

#SUBDIRS(examples Lesson1 Lesson2 Lesson3 Lesson4 Lesson5 Lesson6)
ADD_SUBDIRECTORY(examples)
ADD_SUBDIRECTORY(Lesson1)
ADD_SUBDIRECTORY(Lesson2)
ADD_SUBDIRECTORY(Lesson3)
ADD_SUBDIRECTORY(Lesson4)
ADD_SUBDIRECTORY(Lesson5)
ADD_SUBDIRECTORY(Lesson6)


######################################################################################
### 4) Target to copy benchs and parameters file
######################################################################################

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
			  
ADD_CUSTOM_TARGET(benchs DEPENDS
		${MO_BIN_DIR}/tutorial/examples/tsp/benchs/berlin52.tsp
    	${MO_BIN_DIR}/tutorial/examples/tsp/benchs/eil101.tsp
    	${MO_BIN_DIR}/tutorial/examples/tsp/benchs/pr2392.tsp
    	${MO_BIN_DIR}/tutorial/examples/tsp/benchs/rl5915.tsp
    	${MO_BIN_DIR}/tutorial/examples/tsp/benchs/usa13509.tsp
)			  

ADD_CUSTOM_TARGET(lesson1 DEPENDS ${MO_BIN_DIR}/tutorial/Lesson1/param)

ADD_CUSTOM_TARGET(lesson2 DEPENDS ${MO_BIN_DIR}/tutorial/Lesson2/param)

ADD_CUSTOM_TARGET(lesson3 DEPENDS ${MO_BIN_DIR}/tutorial/Lesson3/param)

ADD_CUSTOM_TARGET(lesson4 DEPENDS ${MO_BIN_DIR}/tutorial/Lesson4/param)

ADD_CUSTOM_TARGET(lesson5 DEPENDS ${MO_BIN_DIR}/tutorial/Lesson5/param)

ADD_CUSTOM_TARGET(lesson6 DEPENDS ${MO_BIN_DIR}/tutorial/Lesson6/param)

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
