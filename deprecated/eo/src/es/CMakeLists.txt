######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Define the es and cma targets
######################################################################################

SET(ES_LIB_OUTPUT_PATH ${EO_BINARY_DIR}/lib)
SET(CMA_LIB_OUTPUT_PATH ${EO_BINARY_DIR}/lib)

SET(LIBRARY_OUTPUT_PATH ${ES_LIB_OUTPUT_PATH}) # the same output for the two libs

SET(ES_SOURCES
  make_algo_scalar_es.cpp
  make_algo_scalar_real.cpp
  make_checkpoint_es.cpp
  make_checkpoint_real.cpp
  make_continue_es.cpp
  make_continue_real.cpp
  make_genotype_es.cpp
  make_genotype_real.cpp
  make_op_es.cpp
  make_op_real.cpp
  make_pop_es.cpp
  make_pop_real.cpp
  make_run_es.cpp
  make_run_real.cpp
  )

SET(CMA_SOURCES
  eig.cpp
  CMAState.cpp
  CMAParams.cpp
  )

ADD_LIBRARY(es STATIC ${ES_SOURCES})
INSTALL(TARGETS es ARCHIVE DESTINATION local/${LIB} COMPONENT libraries)

ADD_LIBRARY(cma STATIC ${CMA_SOURCES})
INSTALL(TARGETS cma ARCHIVE DESTINATION local/${LIB} COMPONENT libraries)

######################################################################################
### 3) Optionnal
######################################################################################

SET(ES_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(es PROPERTIES VERSION "${ES_VERSION}")

SET(CMA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(cma PROPERTIES VERSION "${CMA_VERSION}")

######################################################################################
