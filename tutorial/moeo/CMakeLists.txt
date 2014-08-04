######################################################################################
### 0) Include MO sources
######################################################################################

#include_directories(${MO_SRC_DIR}/src)

######################################################################################
### 1) Define flowshop paths
######################################################################################

set(FLOWSHOP_SRC_DIR ${MOEO_TUTORIAL_DIR}/examples/flowshop)
set(FLOWSHOP_BINARY_DIR ${MOEO_BIN_DIR}/tutorial/examples/flowshop)

######################################################################################
### 2) Include lessons subdirectories
######################################################################################

add_subdirectory(examples)
add_subdirectory(Lesson1)
add_subdirectory(Lesson2)
add_subdirectory(Lesson3)
add_subdirectory(Lesson4)
