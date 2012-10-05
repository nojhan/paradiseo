######################################################################################
### 0) Copy the ESEA.param and RealEA.param files in the build directory for an easy use.
######################################################################################

EXECUTE_PROCESS(
   	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${EO_SOURCE_DIR}/tutorial/Lesson4/ESEA.param
		${EO_BINARY_DIR}/tutorial/Lesson4/ESEA.param
	COMMAND ${CMAKE_COMMAND} -E copy_if_different
		${EO_SOURCE_DIR}/tutorial/Lesson4/RealEA.param
		${EO_BINARY_DIR}/tutorial/Lesson4/RealEA.param
)

##############
# OLD_TARGETS
##############
#ADD_CUSTOM_TARGET(param DEPENDS ${EO_SOURCE_DIR}/tutorial/Lesson4/ESEA.param)
#ADD_CUSTOM_COMMAND(
#    TARGET param
#    POST_BUILD
#    COMMAND ${CMAKE_COMMAND}
#    ARGS -E copy_if_different
#     ${EO_SOURCE_DIR}/tutorial/Lesson4/ESEA.param
#     ${EO_BINARY_DIR}/tutorial/Lesson4)
#ADD_CUSTOM_TARGET(param DEPENDS ${EO_SOURCE_DIR}/tutorial/Lesson4/RealEA.param)
#ADD_CUSTOM_COMMAND(
#    TARGET param
#    POST_BUILD
#    COMMAND ${CMAKE_COMMAND}
#    ARGS -E copy_if_different
#     ${EO_SOURCE_DIR}/tutorial/Lesson4/RealEA.param
#     ${EO_BINARY_DIR}/tutorial/Lesson4)

######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src/es)
INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src/utils)
INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src/ga)

######################################################################################
### 2) Specify where CMake can find the libraries
######################################################################################

IF(NOT WIN32 OR CYGWIN)
  LINK_DIRECTORIES(${EO_BINARY_DIR}/lib)
ENDIF(NOT WIN32 OR CYGWIN)

# especially for Visual Studio
IF(WIN32 AND NOT CYGWIN)
  	LINK_DIRECTORIES(${EO_BINARY_DIR}\\lib\\${CMAKE_BUILD_TYPE})
ENDIF(WIN32 AND NOT CYGWIN)

######################################################################################
### 3) Define your targets
######################################################################################

# no matter what is the OS, hopefully
ADD_EXECUTABLE(BitEA BitEA.cpp)
ADD_EXECUTABLE(RealEA RealEA.cpp)
ADD_EXECUTABLE(ESEA ESEA.cpp)

#ADD_DEPENDENCIES(BitEA es ga eo eoutils)
#ADD_DEPENDENCIES(RealEA es ga eo eoutils)
#ADD_DEPENDENCIES(ESEA es ga eo eoutils)

######################################################################################
### 4) Optionnal
######################################################################################

SET(BITEA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(BitEA PROPERTIES VERSION "${BITEA_VERSION}")

SET(REALEA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(RealEA PROPERTIES VERSION "${REALEA_VERSION}")

SET(ESEA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(ESEA PROPERTIES VERSION "${ESEA_VERSION}")

######################################################################################
### 5) Link the librairies for the targets
######################################################################################

TARGET_LINK_LIBRARIES(BitEA es ga eo eoutils)
TARGET_LINK_LIBRARIES(RealEA es ga eo eoutils)
TARGET_LINK_LIBRARIES(ESEA es ga eo eoutils)

######################################################################################
### 6) Configure project installation paths
######################################################################################

INSTALL(TARGETS BitEA RUNTIME DESTINATION local/share${INSTALL_SUB_DIR}/eo/examples/Lesson4 COMPONENT examples)
INSTALL(TARGETS RealEA RUNTIME DESTINATION local/share${INSTALL_SUB_DIR}/eo/examples/Lesson4 COMPONENT examples)
INSTALL(TARGETS ESEA RUNTIME DESTINATION local/share${INSTALL_SUB_DIR}/eo/examples/Lesson4 COMPONENT examples)

######################################################################################
