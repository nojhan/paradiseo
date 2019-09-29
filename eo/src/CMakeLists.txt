######################################################################################
### 1) Include the sources
######################################################################################

include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Define the eo target
######################################################################################

set(EO_LIB_OUTPUT_PATH ${EO_BIN_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${EO_LIB_OUTPUT_PATH})

set(EO_SOURCES
  eoFunctorStore.cpp
  eoPersistent.cpp
  eoPrintable.cpp
  eoCtrlCContinue.cpp
  eoScalarFitnessAssembled.cpp
  eoSIGContinue.cpp
  )

add_library(eo STATIC ${EO_SOURCES})

######################################################################################
### 3) Optionnal: define your target(s)'s version: no effect for windows
######################################################################################

set(EO_VERSION ${GLOBAL_VERSION})
set_target_properties(eo PROPERTIES VERSION "${EO_VERSION}")

install(TARGETS eo EXPORT paradiseo-targets ARCHIVE DESTINATION ${LIB} COMPONENT libraries)

file(GLOB HDRS *.h eo)
install(FILES ${HDRS} DESTINATION include${INSTALL_SUB_DIR}/eo COMPONENT headers)

install(DIRECTORY do es ga gp other utils
        DESTINATION include${INSTALL_SUB_DIR}/eo
        COMPONENT headers
        FILES_MATCHING PATTERN "*.h" PATTERN "checkpointing" PATTERN external_eo
        )

######################################################################################
### 4) Where must cmake go now ?
######################################################################################

add_subdirectory(es)
add_subdirectory(ga)
add_subdirectory(utils)
#add_subdirectory(serial)

if(ENABLE_PYEO)
  add_subdirectory(pyeo)
endif(ENABLE_PYEO)

######################################################################################
