######################################################################################
### 0) Copy the ESEA.param and RealEA.param files in the build directory for an easy use.
######################################################################################

execute_process(
   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${EO_SRC_DIR}/tutorial/Lesson4/ESEA.param
		${EO_BIN_DIR}/tutorial/Lesson4/ESEA.param
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${EO_SRC_DIR}/tutorial/Lesson4/RealEA.param
		${EO_BIN_DIR}/tutorial/Lesson4/RealEA.param
)

##############
# OLD_TARGETS
##############
#add_custom_target(param DEPENDS ${EO_SRC_DIR}/tutorial/Lesson4/ESEA.param)
#add_custom_command(
#    TARGET param
#    POST_BUILD
#    COMMAND ${CMAKE_COMMAND}
#    ARGS -E copy_if_different
#     ${EO_SRC_DIR}/tutorial/Lesson4/ESEA.param
#     ${EO_BIN_DIR}/tutorial/Lesson4)
#add_custom_target(param DEPENDS ${EO_SRC_DIR}/tutorial/Lesson4/RealEA.param)
#add_custom_command(
#    TARGET param
#    POST_BUILD
#    COMMAND ${CMAKE_COMMAND}
#    ARGS -E copy_if_different
#     ${EO_SRC_DIR}/tutorial/Lesson4/RealEA.param
#     ${EO_BIN_DIR}/tutorial/Lesson4)

######################################################################################
### 1) Include the sources
######################################################################################

include_directories(${EO_SRC_DIR}/src)
include_directories(${EO_SRC_DIR}/src/es)
include_directories(${EO_SRC_DIR}/src/utils)
include_directories(${EO_SRC_DIR}/src/ga)

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
add_executable(BitEA BitEA.cpp)
add_executable(RealEA RealEA.cpp)
add_executable(ESEA ESEA.cpp)

#add_dependencies(BitEA es ga eo eoutils)
#add_dependencies(RealEA es ga eo eoutils)
#add_dependencies(ESEA es ga eo eoutils)

######################################################################################
### 4) Optionnal
######################################################################################

set(BITEA_VERSION ${GLOBAL_VERSION})
set_target_properties(BitEA PROPERTIES VERSION "${BITEA_VERSION}")

set(REALEA_VERSION ${GLOBAL_VERSION})
set_target_properties(RealEA PROPERTIES VERSION "${REALEA_VERSION}")

set(ESEA_VERSION ${GLOBAL_VERSION})
set_target_properties(ESEA PROPERTIES VERSION "${ESEA_VERSION}")

######################################################################################
### 5) Link the librairies for the targets
######################################################################################

target_link_libraries(BitEA es ga eo eoutils)
target_link_libraries(RealEA es ga eo eoutils)
target_link_libraries(ESEA es ga eo eoutils)

######################################################################################
### 6) Configure project installation paths
######################################################################################

install(TARGETS BitEA RUNTIME DESTINATION share${INSTALL_SUB_DIR}/eo/examples/Lesson4 COMPONENT examples)
install(TARGETS RealEA RUNTIME DESTINATION share${INSTALL_SUB_DIR}/eo/examples/Lesson4 COMPONENT examples)
install(TARGETS ESEA RUNTIME DESTINATION share${INSTALL_SUB_DIR}/eo/examples/Lesson4 COMPONENT examples)

######################################################################################
