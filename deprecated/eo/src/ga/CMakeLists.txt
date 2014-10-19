######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Define the ga target
######################################################################################

SET(GA_LIB_OUTPUT_PATH ${EO_BINARY_DIR}/lib)
SET(LIBRARY_OUTPUT_PATH ${GA_LIB_OUTPUT_PATH})

SET(GA_SOURCES
  make_algo_scalar_ga.cpp
  make_checkpoint_ga.cpp
  make_continue_ga.cpp
  make_genotype_ga.cpp
  make_op_ga.cpp
  make_pop_ga.cpp
  make_run_ga.cpp
  )

ADD_LIBRARY(ga STATIC ${GA_SOURCES})
INSTALL(TARGETS ga ARCHIVE DESTINATION local/${LIB} COMPONENT libraries)

######################################################################################
### 3) Optionnal
######################################################################################

SET(GA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(ga PROPERTIES VERSION "${GA_VERSION}")

######################################################################################
