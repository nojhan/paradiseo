######################################################################################
### 1) Include the sources
######################################################################################

include_directories(${EO_SRC_DIR}/src)

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

add_executable(BinaryPSO BinaryPSO.cpp)
add_executable(RealPSO RealPSO.cpp)

######################################################################################
### 4) Optionnal
######################################################################################

set(BINARYPSO_VERSION ${GLOBAL_VERSION})
set_target_properties(BinaryPSO PROPERTIES VERSION "${BINARYPSO_VERSION}")

set(REALPSO_VERSION ${GLOBAL_VERSION})
set_target_properties(RealPSO PROPERTIES VERSION "${REALPSO_VERSION}")

######################################################################################
### 5) Link the librairies for the targets
######################################################################################

target_link_libraries(BinaryPSO eo eoutils)
target_link_libraries(RealPSO eo eoutils)

######################################################################################
### 6) Configure project installation paths
######################################################################################

install(TARGETS BinaryPSO RUNTIME DESTINATION share${INSTALL_SUB_DIR}/eo/examples/Lesson6 COMPONENT examples)
install(TARGETS RealPSO RUNTIME DESTINATION share${INSTALL_SUB_DIR}/eo/examples/Lesson6 COMPONENT examples)

######################################################################################
