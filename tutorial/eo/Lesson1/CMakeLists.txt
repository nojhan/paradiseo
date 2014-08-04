######################################################################################
### 1) Include the sources
######################################################################################

#include_directories(${EO_SRC_DIR}/src)
#include_directories(${EO_SRC_DIR}/src/ga)
#include_directories(${EO_SRC_DIR}/src/utils)

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
add_executable(FirstBitGA FirstBitGA.cpp)
add_executable(FirstRealGA FirstRealGA.cpp)
add_executable(exercise1.3 exercise1.3.cpp)

add_dependencies(FirstBitGA ga eo eoutils)
add_dependencies(FirstRealGA ga eo eoutils)
add_dependencies(exercise1.3 ga eo eoutils)

######################################################################################
### 4) Optionnal
######################################################################################

set(FIRSTBITGA_VERSION ${GLOBAL_VERSION})
set_target_properties(FirstBitGA PROPERTIES VERSION "${FIRSTBITGA_VERSION}")

set(FIRSTREALGA_VERSION ${GLOBAL_VERSION})
set_target_properties(FirstRealGA PROPERTIES VERSION "${FIRSTREALGA_VERSION}")

set(EXERCICE13_VERSION ${GLOBAL_VERSION})
set_target_properties(exercise1.3 PROPERTIES VERSION "${EXERCICE13_VERSION}")

######################################################################################
### 5) Link the librairies for the targets
######################################################################################

target_link_libraries(FirstBitGA ga eo eoutils)
target_link_libraries(FirstRealGA ga eo eoutils)
target_link_libraries(exercise1.3 ga eo eoutils)

######################################################################################
### 6) Configure project installation paths
######################################################################################

install(TARGETS FirstBitGA RUNTIME DESTINATION share/${PROJECT_TAG}/eo/examples/Lesson1 COMPONENT examples)
install(TARGETS FirstRealGA RUNTIME DESTINATION share/${PROJECT_TAG}/eo/examples/Lesson1 COMPONENT examples)
install(TARGETS exercise1.3 RUNTIME DESTINATION share/${PROJECT_TAG}/eo/examples/Lesson1 COMPONENT examples)

######################################################################################
