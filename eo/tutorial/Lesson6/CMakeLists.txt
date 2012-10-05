######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)

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

ADD_EXECUTABLE(BinaryPSO BinaryPSO.cpp)
ADD_EXECUTABLE(RealPSO RealPSO.cpp)

######################################################################################
### 4) Optionnal
######################################################################################

SET(BINARYPSO_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(BinaryPSO PROPERTIES VERSION "${BINARYPSO_VERSION}")

SET(REALPSO_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(RealPSO PROPERTIES VERSION "${REALPSO_VERSION}")

######################################################################################
### 5) Link the librairies for the targets
######################################################################################

TARGET_LINK_LIBRARIES(BinaryPSO eo eoutils)
TARGET_LINK_LIBRARIES(RealPSO eo eoutils)

######################################################################################
### 6) Configure project installation paths
######################################################################################

INSTALL(TARGETS BinaryPSO RUNTIME DESTINATION local/share${INSTALL_SUB_DIR}/eo/examples/Lesson6 COMPONENT examples)
INSTALL(TARGETS RealPSO RUNTIME DESTINATION local/share${INSTALL_SUB_DIR}/eo/examples/Lesson6 COMPONENT examples)

######################################################################################
