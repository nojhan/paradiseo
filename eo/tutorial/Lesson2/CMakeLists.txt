######################################################################################
### 1) Include the sources
######################################################################################

include_directories(${EO_SRC_DIR}/src)
include_directories(${EO_SRC_DIR}/src/ga)
include_directories(${EO_SRC_DIR}/src/utils)

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
add_executable(FirstBitEA FirstBitEA.cpp)
add_executable(FirstRealEA FirstRealEA.cpp)
add_executable(exercise2.3 exercise2.3.cpp)

######################################################################################
### 4) Optionnal
######################################################################################

set(FIRSTBITEA_VERSION ${GLOBAL_VERSION})
set_target_properties(FirstBitEA PROPERTIES VERSION "${FIRSTBITEA_VERSION}")

set(FIRSTREALEA_VERSION ${GLOBAL_VERSION})
set_target_properties(FirstRealEA PROPERTIES VERSION "${FIRSTREALEA_VERSION}")

set(EXERCICE23_VERSION ${GLOBAL_VERSION})
set_target_properties(exercise2.3 PROPERTIES VERSION "${EXERCICE23_VERSION}")

######################################################################################
### 5) Link the librairies for the targets
######################################################################################

target_link_libraries(FirstBitEA ga eo eoutils)
target_link_libraries(FirstRealEA ga eo eoutils)
target_link_libraries(exercise2.3 ga eo eoutils)

######################################################################################
### 6) Configure project installation paths
######################################################################################

install(TARGETS FirstBitEA RUNTIME DESTINATION share${INSTALL_SUB_DIR}/eo/examples/Lesson2 COMPONENT examples)
install(TARGETS FirstRealEA RUNTIME DESTINATION share${INSTALL_SUB_DIR}/eo/examples/Lesson2 COMPONENT examples)
install(TARGETS exercise2.3 RUNTIME DESTINATION share${INSTALL_SUB_DIR}/eo/examples/Lesson2 COMPONENT examples)

######################################################################################
