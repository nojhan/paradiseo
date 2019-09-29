######################################################################################
### 1) Include the sources
######################################################################################

include_directories(${EO_SRC_DIR}/src)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Define the target
######################################################################################

set(EDOUTILS_LIB_OUTPUT_PATH ${EO_BIN_DIR}/lib)
set(LIBRARY_OUTPUT_PATH ${EDOUTILS_LIB_OUTPUT_PATH})

file(GLOB SOURCES *.cpp)
add_library(edoutils ${SOURCES})
install(TARGETS edoutils EXPORT paradiseo-targets ARCHIVE DESTINATION ${LIB} COMPONENT libraries)

set(EDOUTILS_VERSION ${GLOBAL_VERSION})
set_target_properties(edoutils PROPERTIES VERSION "${EDOUTILS_VERSION}")

