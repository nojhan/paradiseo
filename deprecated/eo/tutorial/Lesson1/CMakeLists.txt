######################################################################################
### 1) Include the sources
######################################################################################

INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src)
INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src/ga)
INCLUDE_DIRECTORIES(${EO_SOURCE_DIR}/src/utils)

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
ADD_EXECUTABLE(FirstBitGA FirstBitGA.cpp)
ADD_EXECUTABLE(FirstRealGA FirstRealGA.cpp)
ADD_EXECUTABLE(exercise1.3 exercise1.3.cpp)

ADD_DEPENDENCIES(FirstBitGA ga eo eoutils)
ADD_DEPENDENCIES(FirstRealGA ga eo eoutils)
ADD_DEPENDENCIES(exercise1.3 ga eo eoutils)

######################################################################################
### 4) Optionnal
######################################################################################

SET(FIRSTBITGA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(FirstBitGA PROPERTIES VERSION "${FIRSTBITGA_VERSION}")

SET(FIRSTREALGA_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(FirstRealGA PROPERTIES VERSION "${FIRSTREALGA_VERSION}")

SET(EXERCICE13_VERSION ${GLOBAL_VERSION})
SET_TARGET_PROPERTIES(exercise1.3 PROPERTIES VERSION "${EXERCICE13_VERSION}")

######################################################################################
### 5) Link the librairies for the targets
######################################################################################

TARGET_LINK_LIBRARIES(FirstBitGA ga eo eoutils)
TARGET_LINK_LIBRARIES(FirstRealGA ga eo eoutils)
TARGET_LINK_LIBRARIES(exercise1.3 ga eo eoutils)

######################################################################################
### 6) Configure project installation paths
######################################################################################

INSTALL(TARGETS FirstBitGA RUNTIME DESTINATION local/share${INSTALL_SUB_DIR}/eo/examples/Lesson1 COMPONENT examples)
INSTALL(TARGETS FirstRealGA RUNTIME DESTINATION local/share${INSTALL_SUB_DIR}/eo/examples/Lesson1 COMPONENT examples)
INSTALL(TARGETS exercise1.3 RUNTIME DESTINATION local/share${INSTALL_SUB_DIR}/eo/examples/Lesson1 COMPONENT examples)

######################################################################################
