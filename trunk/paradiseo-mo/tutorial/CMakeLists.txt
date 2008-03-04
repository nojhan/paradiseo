
######################################################################################
### 1) Definitions (required for tsp target)
######################################################################################

SET(TSP_SRC_DIR ${ParadisEO-MO_SOURCE_DIR}/tutorial/examples/tsp CACHE PATH "TSP src directory")
SET(TSP_BINARY_DIR ${ParadisEO-MO_BINARY_DIR}/tutorial/examples/tsp CACHE PATH "TSP binary directory")

SET(FUNCTION_SRC_DIR ${ParadisEO-MO_SOURCE_DIR}/tutorial/examples/function CACHE PATH "FUNCTION src directory")
SET(FUNCTION_BINARY_DIR ${ParadisEO-MO_BINARY_DIR}/tutorial/examples/function CACHE PATH "FUNCTION binary directory")

######################################################################################

######################################################################################
### 2) Where must cmake go now ?
######################################################################################

SUBDIRS(examples Lesson1 Lesson2 Lesson3 Lesson4 Lesson5)

######################################################################################
