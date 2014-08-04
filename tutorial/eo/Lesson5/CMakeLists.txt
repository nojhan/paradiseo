######################################################################################
### 1) Include the sources
######################################################################################

#include_directories(${EO_SRC_DIR}/src)
#include_directories(${CMAKE_CURRENT_SOURCE_DIR})

######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

if(NOT WIN32 OR CYGWIN)
  link_directories(${EO_BIN_DIR}/lib)
endif(NOT WIN32 OR CYGWIN)

# especially for Visual Studio
if(WIN32 AND NOT CYGWIN)
  	link_directories(${EO_BIN_DIR}\\lib\\${CMAKE_BUILD_TYPE})
endif(WIN32 AND NOT CYGWIN)

######################################################################################
### 3) Define your targets
######################################################################################

# no matter what is the OS, hopefully
add_executable(OneMaxEA OneMaxEA.cpp)
add_executable(OneMaxLibEA OneMaxLibEA.cpp make_OneMax.cpp)

add_dependencies(OneMaxEA es ga eo eoutils)
add_dependencies(OneMaxLibEA es ga eo eoutils)

######################################################################################
### 4) Optionnal
######################################################################################

set(ONEMAXEA_VERSION ${GLOBAL_VERSION})
set_target_properties(OneMaxEA PROPERTIES VERSION "${ONEMAXEA_VERSION}")

set(ONEMAXLIBEA_VERSION ${GLOBAL_VERSION})
set_target_properties(OneMaxLibEA PROPERTIES VERSION "${ONEMAXLIBEA_VERSION}")

######################################################################################
### 5) Link the librairies for the targets
######################################################################################

target_link_libraries(OneMaxEA es ga eo eoutils)
target_link_libraries(OneMaxLibEA es ga eo eoutils)

######################################################################################
### 6) Configure project installation paths
######################################################################################

install(TARGETS OneMaxEA RUNTIME DESTINATION share/${PROJECT_TAG}/eo/examples/Lesson5 COMPONENT examples)
install(TARGETS OneMaxLibEA RUNTIME DESTINATION share/${PROJECT_TAG}/eo/examples/Lesson5 COMPONENT examples)

######################################################################################
