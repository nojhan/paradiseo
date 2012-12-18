######################################################################################
### 1) Include the sources
######################################################################################

include_directories(${EO_SRC_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Define the ga target
######################################################################################

set(GA_LIB_OUTPUT_PATH ${EO_BIN_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${GA_LIB_OUTPUT_PATH})

set(GA_SOURCES
  make_algo_scalar_ga.cpp
  make_checkpoint_ga.cpp
  make_continue_ga.cpp
  make_genotype_ga.cpp
  make_op_ga.cpp
  make_pop_ga.cpp
  make_run_ga.cpp
  )

add_library(ga STATIC ${GA_SOURCES})
install(TARGETS ga ARCHIVE DESTINATION lib COMPONENT libraries)

file(GLOB HDRS *.h)
install(FILES ${HDRS} DESTINATION include/eo/ga COMPONENT headers)

######################################################################################
### 3) Optionnal
######################################################################################

set(GA_VERSION ${GLOBAL_VERSION})
set_target_properties(ga PROPERTIES VERSION "${GA_VERSION}")

######################################################################################
