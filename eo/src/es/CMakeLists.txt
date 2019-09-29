######################################################################################
### 1) Include the sources
######################################################################################

include_directories(${EO_SRC_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Define the es and cma targets
######################################################################################

set(ES_LIB_OUTPUT_PATH ${EO_BIN_DIR}/lib)
set(CMA_LIB_OUTPUT_PATH ${EO_BIN_DIR}/lib)

set(LIBRARY_OUTPUT_PATH ${ES_LIB_OUTPUT_PATH}) # the same output for the two libs

set(ES_SOURCES
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

set(CMA_SOURCES
  eig.cpp
  CMAState.cpp
  CMAParams.cpp
  )

add_library(es STATIC ${ES_SOURCES})
install(TARGETS es EXPORT paradiseo-targets ARCHIVE DESTINATION ${LIB} COMPONENT libraries)

add_library(cma STATIC ${CMA_SOURCES})
install(TARGETS cma EXPORT paradiseo-targets ARCHIVE DESTINATION ${LIB} COMPONENT libraries)

######################################################################################
### 3) Optionnal
######################################################################################

set(ES_VERSION ${GLOBAL_VERSION})
set_target_properties(es PROPERTIES VERSION "${ES_VERSION}")

set(CMA_VERSION ${GLOBAL_VERSION})
set_target_properties(cma PROPERTIES VERSION "${CMA_VERSION}")

######################################################################################
