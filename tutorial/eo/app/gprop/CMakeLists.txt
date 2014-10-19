######################################################################################
### 1) Include the sources
######################################################################################

#include_directories(${EO_SRC_DIR}/src)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) specify where cmake can find the libraries (mandatory: before 3) )
######################################################################################

link_directories(${EO_BIN_DIR}/lib)

######################################################################################
### 3) define your target(s): just an executable here
######################################################################################

set (GPROP_SOURCES gprop.cpp)

# especially for Visual Studio
if(NOT WIN32 OR CYGWIN)
    add_executable(gprop ${GPROP_SOURCES})
    add_dependencies(gprop eo eoutils)

    target_link_libraries(gprop eo eoutils)

    set(GPROP_VERSION ${GLOBAL_VERSION})
  set_target_properties(gprop PROPERTIES VERSION "${GPROP_VERSION}")
endif(NOT WIN32 OR CYGWIN)

######################################################################################
