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

ADD_SUBDIRECTORY(examples)
ADD_SUBDIRECTORY(Lesson1)
ADD_SUBDIRECTORY(Lesson2)
ADD_SUBDIRECTORY(Lesson3)
ADD_SUBDIRECTORY(Lesson4)
ADD_SUBDIRECTORY(Lesson5)
ADD_SUBDIRECTORY(Lesson6)
######################################################################################
