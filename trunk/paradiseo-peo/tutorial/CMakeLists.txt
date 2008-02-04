#
#####################################################################################
### 1) Definitions (required for tsp target)
######################################################################################

SET(TSP_SRC_DIR ${ParadisEO-PEO_SOURCE_DIR}/tutorial/examples/tsp)
SET(TSP_BINARY_DIR ${ParadisEO-PEO_BINARY_DIR}/tutorial/examples/tsp)
SET(FLOWSHOP_SRC_DIR "${MOEO_SRC_DIR}/tutorial/examples/flowshop" CACHE PATH "Flowshop source directory" FORCE)
SET(FLOWSHOP_BIN_DIR "${MOEO_BIN_DIR}/tutorial/examples/flowshop" CACHE PATH "Flowshop binary directory" FORCE)

######################################################################################

######################################################################################
### 2) Where must cmake go now ?
######################################################################################

SUBDIRS(examples Lesson1 Lesson2 Lesson3 Lesson4 Lesson5 Lesson6 Lesson7 Lesson8)

######################################################################################
