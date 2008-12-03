
######################################################################################
### 1) Definitions (required for tsp target)
######################################################################################

SET(TSP_SRC_DIR ${MO_SRC_DIR}/tutorial/examples/tsp CACHE PATH "TSP src directory")
SET(TSP_BIN_DIR ${MO_BIN_DIR}/tutorial/examples/tsp CACHE PATH "TSP binary directory")

SET(FUNCTION_SRC_DIR ${MO_SRC_DIR}/tutorial/examples/function CACHE PATH "FUNCTION src directory")
SET(FUNCTION_BIN_DIR ${MO_BIN_DIR}/tutorial/examples/function CACHE PATH "FUNCTION binary directory")

######################################################################################

######################################################################################
### 2) Where must cmake go now ?
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


ADD_CUSTOM_TARGET(install DEPENDS ${MO_SRC_DIR}/tutorial/examples/tsp/benchs
			  	  ${MO_SRC_DIR}/tutorial/Lesson1/param
			  	  ${MO_SRC_DIR}/tutorial/Lesson2/param
			  	  ${MO_SRC_DIR}/tutorial/Lesson3/param 
				  ${MO_SRC_DIR}/tutorial/Lesson4/param 
				  ${MO_SRC_DIR}/tutorial/Lesson5/param 
				  ${MO_SRC_DIR}/tutorial/Lesson6/param )

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_directory 
      ${MO_SRC_DIR}/tutorial/examples/tsp/benchs
      ${MO_BIN_DIR}/tutorial/examples/tsp/benchs)

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${MO_SRC_DIR}/tutorial/Lesson1/param
      ${MO_BIN_DIR}/tutorial/Lesson1)

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${MO_SRC_DIR}/tutorial/Lesson2/param
      ${MO_BIN_DIR}/tutorial/Lesson2)

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${MO_SRC_DIR}/tutorial/Lesson3/param
      ${MO_BIN_DIR}/tutorial/Lesson3)

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${MO_SRC_DIR}/tutorial/Lesson4/param
      ${MO_BIN_DIR}/tutorial/Lesson4)

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${MO_SRC_DIR}/tutorial/Lesson5/param
      ${MO_BIN_DIR}/tutorial/Lesson5)

ADD_CUSTOM_COMMAND(
    TARGET install
    POST_BUILD
    COMMAND ${CMAKE_COMMAND}
    ARGS -E copy_if_different
      ${MO_SRC_DIR}/tutorial/Lesson6/param
      ${MO_BIN_DIR}/tutorial/Lesson6)
