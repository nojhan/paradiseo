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
add_executable(SecondBitEA SecondBitEA.cpp)
add_executable(SecondRealEA SecondRealEA.cpp)
add_executable(exercise3.1 exercise3.1.cpp)

add_dependencies(SecondBitEA ga eoutils eo)
add_dependencies(SecondRealEA ga eoutils eo)
add_dependencies(exercise3.1 ga eoutils eo)

######################################################################################
### 4) Optionnal
######################################################################################

set(SECONDBITEA_VERSION ${GLOBAL_VERSION})
set_target_properties(SecondBitEA PROPERTIES VERSION "${SECONDBITEA_VERSION}")

set(SECONDREALEA_VERSION ${GLOBAL_VERSION})
set_target_properties(SecondRealEA PROPERTIES VERSION "${SECONDREALEA_VERSION}")

set(EXERCICE31_VERSION ${GLOBAL_VERSION})
set_target_properties(exercise3.1 PROPERTIES VERSION "${EXERCICE31_VERSION}")

######################################################################################
### 5) Link the librairies for the targets
######################################################################################

target_link_libraries(SecondBitEA ga eoutils eo)
target_link_libraries(SecondRealEA ga eoutils eo)
target_link_libraries(exercise3.1 ga eoutils eo)

######################################################################################
### 6) Configure project installation paths
######################################################################################

install(TARGETS SecondBitEA RUNTIME DESTINATION local/share${INSTALL_SUB_DIR}/eo/examples/Lesson3 COMPONENT examples)
install(TARGETS SecondRealEA RUNTIME DESTINATION local/share${INSTALL_SUB_DIR}/eo/examples/Lesson3 COMPONENT examples)
install(TARGETS exercise3.1 RUNTIME DESTINATION local/share${INSTALL_SUB_DIR}/eo/examples/Lesson3 COMPONENT examples)

######################################################################################
